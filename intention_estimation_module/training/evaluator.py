# ---working.
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

# Target directory path where your new train.py saves its optimized weights
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models/fine_tuned_classifier")

# Safety Check: If relative path mapping fails, fallback to absolute server paths
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "/home/sax023/LanguageBind/intention_estimation_module/models/fine_tuned_classifier"

print(f"[Evaluator] Actively loading configuration matrix from: {MODEL_PATH}")

# 1. DYNAMIC CONFIG DETECTION
# This forces the script to read the exact class matrix sizes stored by your train run
config = AutoConfig.from_pretrained(MODEL_PATH)
print(f"[Evaluator] Successfully detected checkpoint with {config.num_labels} classification fields.")

# 2. SEAMLESS TOKENIZER INITIALIZATION
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)

# 3. CONFIGURE SHAPE-ALIGNED BASE LAYERS
# Passing the config explicitly prevents the [2, 1024] fallback layer creation
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    config=config,
    use_safetensors=False
)

model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def predict(query):
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()
        
        # Guard against zero-dimensional squeezing on single batch sizes
        if probs.dim() == 0:
            confidence = probs
            idx = torch.tensor(0).to(device)
        else:
            confidence, idx = torch.max(probs, dim=0)

    # Convert the token ID cleanly back into your pipe-delimited string layout
    label_str = model.config.id2label[idx.item()]
    fields = label_str.split("|")
    
    # Return a structured, unpacked result dictionary
    return {
        "task_type": fields[0] if len(fields) > 0 else "unknown",
        "output_modality": fields[1] if len(fields) > 1 else "text",
        "complexity": fields[2] if len(fields) > 2 else "simple",
        "temporal_context": fields[3] if len(fields) > 3 else "none",
        "spatial_context": fields[4] if len(fields) > 4 else "none",
        "confidence": float(confidence.item())
    }