#Code that will monitor, deploy any dynamic model using FastAPI
import os
import time
import torch
import mlflow
import mlflow.pytorch
import onnx
import subprocess
import asyncio
import logging
from fastapi import FastAPI, UploadFile, File, Form, Response, Depends, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
from transformers import AutoTokenizer, AutoModelForMaskedLM
from PIL import Image
from torchvision import transforms
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi.responses import JSONResponse
import tritonclient.http as httpclient

# Initialize FastAPI app
app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up Prometheus monitoring
Instrumentator().instrument(app).expose(app)

# Set up OAuth2 for security
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Initialize models and tokenizers
models = {}
model_paths = {
    "bert-base-uncased": ("bert", "bert-base-uncased"),
    "resnet18": ("vision", "resnet18"),
}

# Define a simple user database for authentication
fake_users_db = {
    "alice": {
        "username": "alice",
        "hashed_password": "fakehashedpassword",
    }
}

def fake_hash_password(password: str):
    return "fakehashed" + password

# Function to get current user
async def get_current_user(token: str = Depends(oauth2_scheme)):
    user = fake_users_db.get(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    return user

# Load initial models
def load_initial_models():
    global models
    models["bert-base-uncased"] = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    models["resnet18"] = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True).eval()

load_initial_models()

# Image transformation setup
image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to dynamically load a model
@app.post("/load_model/")
async def load_model(model_name: str, user: dict = Depends(get_current_user)):
    if model_name not in model_paths:
        return {"error": "Invalid model name"}
    model_type, model_id = model_paths[model_name]
    if model_type == "bert":
        models[model_name] = AutoModelForMaskedLM.from_pretrained(model_id)
    elif model_type == "vision":
        models[model_name] = torch.hub.load('pytorch/vision:v0.10.0', model_id, pretrained=True).eval()
    return {"message": f"{model_name} loaded successfully"}

# Function to dynamically unload a model
@app.post("/unload_model/")
async def unload_model(model_name: str, user: dict = Depends(get_current_user)):
    if model_name in models:
        del models[model_name]
        return {"message": f"{model_name} unloaded successfully"}
    else:
        return {"error": "Model not found"}

# Generate text using BERT model
def generate_text(input_text: str, model):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    inputs = tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits

# Process image using ResNet model
def process_image(image: Image.Image, model):
    image_tensor = image_transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
    return outputs

# Endpoint to get prediction
@app.post("/predict/")
async def predict(model_type: str = Form(...), text: str = Form(None), image: UploadFile = File(None), user: dict = Depends(get_current_user)):
    start_time = time.time()
    if model_type == "text" and text:
        if "bert-base-uncased" not in models:
            return {"error": "Text model not loaded"}
        result = generate_text(text, models["bert-base-uncased"])
        metric = {"type": "text", "result": result.tolist()}
    elif model_type == "image" and image:
        if "resnet18" not in models:
            return {"error": "Image model not loaded"}
        img = Image.open(image.file)
        result = process_image(img, models["resnet18"])
        metric = {"type": "image", "result": result.tolist()}
    else:
        return {"error": "Invalid input"}

    end_time = time.time()
    latency = end_time - start_time
    throughput = 1 / latency  # Simplified example, adjust based on your metric definitions

    metric["latency"] = latency
    metric["throughput"] = throughput

    return metric

# OAuth2 token endpoint
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user_dict = fake_users_db.get(form_data.username)
    if not user_dict or user_dict["hashed_password"] != fake_hash_password(form_data.password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    return {"access_token": form_data.username, "token_type": "bearer"}

# Root endpoint
@app.get("/")
async def read_root():
    return {"message": "Welcome to the API!"}

# Favicon endpoint
@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

# Custom exception handler
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Exception: {exc}")
    return JSONResponse(status_code=500, content={"detail": "An internal server error occurred"})

# Global variables to store model state for Triton
triton_model_names = set()  # Using a set to ensure unique model names
triton_server_process = None

class DeployModelsRequest(BaseModel):
    models: List[str] = Field(..., example=["mnist_model_onnx", "mnist_model_openvino"])

@app.post("/deploy_triton/")
async def deploy_models(request_body: DeployModelsRequest):
    models = request_body.models
    global triton_server_process, triton_model_names
    if triton_server_process:
        triton_server_process.kill()

    required_models = {"mnist_model_onnx", "mnist_model_openvino", "mnist_model_pytorch", "mnist_model_tensorflow"}
    if not all(model in required_models for model in models):
        return JSONResponse(status_code=422, content={"detail": "Invalid model names. Use the exact model names."})

    docker_run_command = [
        'docker', 'run', '--shm-size=256m', '--rm',
        '-p8000:8000', '-p8001:8001', '-p8002:8002',
        '-e', 'TRITON_SERVER_CPU_ONLY=1',
        '-v', f'{os.getcwd()}:/workspace/',
        '-v', f'{os.getcwd()}/model_repository:/models',
        'nvcr.io/nvidia/tritonserver:24.04-py3',
        'tritonserver', '--model-repository=/models',
        '--model-control-mode=explicit'
    ]

    for model in models:
        docker_run_command.extend(['--load-model', model])

    try:
        triton_server_process = subprocess.Popen(' '.join(docker_run_command), shell=True)
        triton_model_names = set(models)  # Clear and set the new models
        return {"message": "Models deployed successfully"}
    except Exception as e:
        logger.error(f"Error deploying models: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": "Failed to deploy models"})

@app.post("/infer_triton/")
async def infer(background_tasks: BackgroundTasks, model_name: str = Query(...)):
    if model_name not in triton_model_names:
        raise HTTPException(status_code=404, detail="Model not found")

    await background_tasks.add_task(run_inference, model_name)
    return {"message": "Inference started"}

async def run_inference(model_name: str):
    try:
        url = "localhost:8000"  # Update to the correct port
        client = httpclient.InferenceServerClient(url=url)

        client_file_map = {
            "mnist_model_onnx": "./client/client_onnx.py",
            "mnist_model_openvino": "./client/client_openvino.py",
            "mnist_model_pytorch": "./client/client_pytorch.py",
            "mnist_model_tensorflow": "./client/client_tensorflow.py"
        }

        if model_name not in client_file_map:
            raise ValueError("Invalid model name")

        client_file = client_file_map[model_name]

        # Log the start of inference
        logger.info(f"Starting inference for model: {model_name}")

        # Execute the appropriate client file asynchronously
        process = await asyncio.create_subprocess_exec('python3', client_file)
        stdout, stderr = await process.communicate()

        # Log the output and errors
        if stdout:
            logger.info(f"Inference output: {stdout.decode()}")
        if stderr:
            logger.error(f"Inference error: {stderr.decode()}")

    except Exception as e:
        logger.error(f"Exception during inference: {str(e)}")

@app.get("/results_triton/")
async def get_results():
    try:
        with open("backend/results.txt", "r") as f:
            results = f.read()
        return {"results": results}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Results file not found")
    except Exception as e:
        logger.error(f"Error reading results file: {str(e)}")
        raise HTTPException(status_code=500, detail="Error reading results file")

# Simplified endpoint to test basic functionality
@app.get("/test/")
async def test():
    return {"message": "Test endpoint is working"}

@app.post("/test_infer/")
async def test_infer(model_name: str = Query(...)):
    return {"model_name": model_name}

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
