import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForMaskedLM
import time
import mlflow
import mlflow.pytorch
import torch.onnx
import onnx

# Set Kaggle API credentials using environment variables
os.environ['KAGGLE_USERNAME'] = 'siddhirajpurohit'
os.environ['KAGGLE_KEY'] = '4bbd59494b9a28037a46efe631bc1b54'

# Initialize Kaggle API
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

# Download the dataset
dataset = 'thedevastator/udemy-courses-revenue-generation-and-course-anal'
api.dataset_download_files(dataset, path='.', unzip=True)

# Paths to the CSV files
csv_files = [
    '3.1-data-sheet-udemy-courses-business-courses.csv',
    '3.1-data-sheet-udemy-courses-design-courses.csv',
    '3.1-data-sheet-udemy-courses-music-courses.csv',
    '3.1-data-sheet-udemy-courses-web-development.csv',
    'Entry Level Project Sheet - 3.1-data-sheet-udemy-courses-web-development.csv'
]

# Load and preprocess the dataset
dataframes = [pd.read_csv(file) for file in csv_files]
df = pd.concat(dataframes, ignore_index=True)

# Create text data using 'course_title' and 'subject'
df['text'] = df['course_title'].fillna('') + " - " + df['subject'].fillna('')

# Ensure all texts are strings
texts = df['text'].astype(str).tolist()

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

# Create a custom dataset class
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length")
        return inputs.input_ids.squeeze(), inputs.attention_mask.squeeze()

# Split the dataset and create dataloaders
train_texts, test_texts = train_test_split(texts, test_size=0.2, random_state=42)
max_length = 128
batch_size = 8
train_dataset = TextDataset(train_texts, tokenizer, max_length)
test_dataset = TextDataset(test_texts, tokenizer, max_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Set up the optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
total_steps = len(train_loader) * 3  # number of epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# Define the training loop
def train_model(model, dataloader, optimizer, scheduler, epochs=3):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (input_ids, attention_mask) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}")
    return model

# Train the model
trained_model = train_model(model, train_loader, optimizer, scheduler)

# Define the evaluation function
def evaluate_model(model, dataloader):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    start_time = time.time()
    
    with torch.no_grad():
        for input_ids, attention_mask in dataloader:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == input_ids).sum().item()
            total_predictions += input_ids.numel()
    
    end_time = time.time()
    
    accuracy = correct_predictions / total_predictions
    latency = (end_time - start_time) / len(dataloader.dataset)
    throughput = len(dataloader.dataset) / (end_time - start_time)
    
    return accuracy, latency, throughput

# Evaluate the model
accuracy, latency, throughput = evaluate_model(trained_model, test_loader)

# Print the results
print(f"Accuracy: {accuracy:.4f}")
print(f"Latency: {latency:.8f} seconds per token")
print(f"Throughput: {throughput:.2f} tokens per second")

# Export to ONNX
def export_to_onnx(model, file_path):
    dummy_input = torch.randint(0, tokenizer.vocab_size, (1, max_length))  # Adjust according to your model input
    torch.onnx.export(model, dummy_input, file_path, opset_version=14)

export_to_onnx(trained_model, 'bert_model.onnx')

# Log model with MLflow
def log_model_with_mlflow(model, onnx_file_path="bert_model.onnx"):
    mlflow.set_experiment("BERT_Text_Generation_Experiment")
    with mlflow.start_run() as run:
        mlflow.pytorch.log_model(model, "pytorch_model")
        export_to_onnx(model, onnx_file_path)
        mlflow.onnx.log_model(onnx_model=onnx.load(onnx_file_path), artifact_path="onnx_model")

log_model_with_mlflow(trained_model)
print("Model training, evaluation, conversion to ONNX, and logging with MLflow complete.")
