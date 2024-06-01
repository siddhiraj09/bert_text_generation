import argparse
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
model.eval()

def generate_text(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=inputs.input_ids)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        predicted_tokens = tokenizer.decode(predictions[0], skip_special_tokens=True)
    return predicted_tokens

def main():
    parser = argparse.ArgumentParser(description="Text generation using BERT model")
    parser.add_argument('--text', type=str, required=True, help='Input text for text generation')
    args = parser.parse_args()
    
    generated_text = generate_text(args.text)
    print(f"Generated Text: {generated_text}")

if __name__ == '__main__':
    main()
