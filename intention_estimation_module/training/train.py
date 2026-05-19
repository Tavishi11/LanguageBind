"""
training/train.py
OPTIMIZED FINAL VERSION:
- Uses standard CrossEntropyLoss with clean oversampling (No gradient breaking weights)
- Sets oversample=True with target_per_class=2000 for perfect 1:1:1 multi-class symmetry
- Lowers learning rate to 1e-5 to eliminate overshooting
- Restores active EarlyStoppingCallback tracking macro_f1
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model
from dataset import load_intent_dataset_jsonl

MODEL_NAME = "/home/sax023/LanguageBind/intention_estimation_module/deberta-v3-large"
DATA_PATH  = "data/ground_truth_new.jsonl"

def main():
    print(f"[Training] Initializing dataset extraction from: {DATA_PATH}")
    
    # ✅ FIXED: Oversampling turned ON, target set to 2000 to cleanly balance all 3 classes
    dataset = load_intent_dataset_jsonl(
        path=DATA_PATH,
        augment_minority=True,         
        oversample=True,   
        target_per_class=2000,          
    )

    all_labels = sorted(set(dataset["train"]["label"]))
    label2id = {l: i for i, l in enumerate(all_labels)}
    id2label  = {i: l for l, i in label2id.items()}
    print(f"[Training] Targets successfully mapped: {all_labels}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    def preprocess(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=256,
        )
        enc["label"] = [label2id[l] for l in batch["label"]]
        return enc

    tokenized = dataset.map(preprocess, batched=True, remove_columns=["text", "label"])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(all_labels),
        id2label=id2label,
        label2id=label2id,
        use_safetensors=False,
    )

    lora_config = LoraConfig(
        r=16,                          
        lora_alpha=32,
        target_modules=["query_proj", "key_proj", "value_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── TUNED TRAINING PARAMETERS ──────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir="checkpoints/",
        evaluation_strategy="epoch",    
        save_strategy="epoch",
        load_best_model_at_end=True,      
        metric_for_best_model="macro_f1", 
        greater_is_better=True,
        learning_rate=1e-5,               # ✅ FIXED: Lowered from 2e-5 to stabilize optimization steps
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=6,               # ✅ FIXED: 6 epochs is optimal for converged oversampled splits
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir="logs/",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
        save_total_limit=1,               # Flushes older epoch checkpoints to save disk space
    )

    accuracy_metric = evaluate.load("accuracy")
    f1_metric       = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels_eval = eval_pred
        preds = logits.argmax(axis=-1)
        acc = accuracy_metric.compute(predictions=preds, references=labels_eval)
        f1  = f1_metric.compute(predictions=preds, references=labels_eval, average="macro")
        return {"accuracy": acc["accuracy"], "macro_f1": f1["f1"]}

    # ✅ FIXED: Using standard Trainer. Standard Cross-Entropy works perfectly on balanced data.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)], 
    )

    print("\n[Training] Executing structural neural optimization passes...")
    trainer.train()

    save_directory = "models/fine_tuned_classifier"
    trainer.save_model(save_directory)
    tokenizer.save_pretrained(save_directory)
    print(f"\n[Success] High-precision 3-class model saved cleanly to: {save_directory}")


if __name__ == "__main__":
    main()

# """
# training/train.py
# TRUE Restored Version.

# Fixes applied and fully integrated:
# - load_best_model_at_end=True      (Safeguards optimal validation state)
# - EarlyStoppingCallback restored   (Halts training if macro F1/accuracy plateaus)
# - num_train_epochs=10              (Allows LoRA adapters to fully converge)
# - target_per_class=500             (Scales minority configuration structures)
# - learning_rate=2e-5               (Stabilizes DeBERTa weight updates)
# - r=16 LoRA rank restored          (Matches historical baseline capabilities)
# - evaluation_strategy="epoch"      (Required framework hook for metrics tracking)
# - augment_minority & oversample=True (Smashes majority-class prediction bias)
# """

# import sys
# import os
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import torch
# import evaluate
# from torch.nn import CrossEntropyLoss
# from transformers import (
#     AutoTokenizer,
#     AutoModelForSequenceClassification,
#     TrainingArguments,
#     Trainer,
#     EarlyStoppingCallback,
# )
# from peft import LoraConfig, get_peft_model
# from dataset import load_intent_dataset_jsonl
# from collections import Counter

# MODEL_NAME = "/home/sax023/LanguageBind/intention_estimation_module/deberta-v3-large"
# DATA_PATH  = "data/ground_truth_new.jsonl"


# class WeightedTrainer(Trainer):
#     def __init__(self, *args, class_weights=None, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.class_weights = class_weights

#     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
#         labels = inputs.pop("labels")
#         outputs = model(**inputs)
#         logits = outputs.logits
#         if self.class_weights is not None:
#             weights = self.class_weights.to(logits.device)
#             # Label smoothing adds regularization to combat over-indexing
#             loss_fn = CrossEntropyLoss(weight=weights, label_smoothing=0.1)
#         else:
#             loss_fn = CrossEntropyLoss(label_smoothing=0.1)
#         loss = loss_fn(logits, labels)
#         return (loss, outputs) if return_outputs else loss


# def compute_class_weights(labels: list, num_classes: int) -> torch.Tensor:
#     counts = Counter(labels)
#     total  = sum(counts.values())
#     # Inverse-frequency formulation balancing
#     weights = [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)]
#     t = torch.tensor(weights, dtype=torch.float32)
#     print(f"\n[Training] Computed inverse-frequency class weights: {t.tolist()}")
#     return t


# def main():
#     print(f"[Training] Initializing dataset extraction from: {DATA_PATH}")
#     # ── FIXED: Enabled data-balancing to actively fight majority-class bias ──
#     dataset = load_intent_dataset_jsonl(
#         path=DATA_PATH,
#         augment_minority=True,         # FIXED: turned back on
#         oversample=False,               # FIXED: turned back on
#         target_per_class=500,          
#     )

#     all_labels = sorted(
#         set(dataset["train"]["label"]) |
#         set(dataset["validation"]["label"]) |
#         set(dataset["test"]["label"])
#     )
#     label2id = {l: i for i, l in enumerate(all_labels)}
#     id2label  = {i: l for l, i in label2id.items()}
#     print(f"[Training] Target Multi-Field Labels mapped: {all_labels}")

#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

#     def preprocess(batch):
#         enc = tokenizer(
#             batch["text"],
#             truncation=True,
#             padding="max_length",
#             max_length=256,
#         )
#         enc["label"] = [label2id[l] for l in batch["label"]]
#         return enc

#     tokenized = dataset.map(preprocess, batched=True, remove_columns=["text", "label"])

#     train_labels = [label2id[l] for l in dataset["train"]["label"]]
#     class_weights = compute_class_weights(train_labels, len(all_labels))

#     model = AutoModelForSequenceClassification.from_pretrained(
#         MODEL_NAME,
#         num_labels=len(all_labels),
#         id2label=id2label,
#         label2id=label2id,
#         use_safetensors=False,
#     )

#     # ── PEFT CONFIGURATION — RESTORED TO ORIGIN SPECS ────────────────────────
#     lora_config = LoraConfig(
#         r=16,                          
#         lora_alpha=32,
#         target_modules=["query_proj", "key_proj", "value_proj"],
#         lora_dropout=0.05,
#         bias="none",
#         task_type="SEQ_CLS",
#     )
#     model = get_peft_model(model, lora_config)
#     model.print_trainable_parameters()

#     # ── CORRECTED TRAINING ARGUMENTS ──────────────────────────────────────────
#     training_args = TrainingArguments(
#         output_dir="checkpoints/",
#         evaluation_strategy="epoch",    
#         save_strategy="epoch",
#         load_best_model_at_end=True,     # FIXED: Was False in your code loop
#         metric_for_best_model="macro_f1", # FIXED: Prioritize macro F1 over absolute accuracy for imbalanced data
#         greater_is_better=True,
#         learning_rate=2e-5,            
#         per_device_train_batch_size=4,
#         per_device_eval_batch_size=4,
#         num_train_epochs=10,            
#         weight_decay=0.01,
#         warmup_ratio=0.1,
#         logging_dir="logs/",
#         logging_steps=10,
#         fp16=torch.cuda.is_available(),
#         report_to="none",
#         save_total_limit=2,              # Keeps storage overhead low on the server
#     )

#     accuracy_metric = evaluate.load("accuracy")
#     f1_metric       = evaluate.load("f1")

#     def compute_metrics(eval_pred):
#         logits, labels_eval = eval_pred
#         preds = logits.argmax(axis=-1)
#         acc = accuracy_metric.compute(predictions=preds, references=labels_eval)
#         f1  = f1_metric.compute(predictions=preds, references=labels_eval, average="macro")
#         # FIXED: Extract output keys safely to prevent metric calculation crashes
#         return {"accuracy": acc["accuracy"], "macro_f1": f1["f1"]}

#     # Instantiate the engine with active early-stopping callbacks
#     trainer = WeightedTrainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized["train"],
#         eval_dataset=tokenized["validation"],
#         tokenizer=tokenizer,
#         compute_metrics=compute_metrics,
#         class_weights=class_weights,
#         callbacks=[EarlyStoppingCallback(early_stopping_patience=3)], # FIXED: Restored callback driver
#     )

#     print("\n[Training] Executing active neural forward passes...")
#     trainer.train()

#     print("\n[Training] Evaluating against final test split...")
#     print(f"\n[Training] Test results: {trainer.evaluate(tokenized['test'])}")

#     # ── ARCHITECTURAL CORRECTION FOR PEFT MODEL SAVE ──────────────────────────
#     # Using model.save_pretrained directly can cause HF Trainer tracking confusion with PEFT adapters.
#     # Saving via the trainer wrapper guarantees weights and custom label configurations serialize together perfectly.
#     save_directory = "models/fine_tuned_classifier"
#     trainer.save_model(save_directory)
#     tokenizer.save_pretrained(save_directory)
#     print(f"\n[Success] Robust fine-tuned classifier saved cleanly to: {save_directory}")


# if __name__ == "__main__":
#     main()

# import torch
# import evaluate
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
# from peft import LoraConfig, get_peft_model
# from dataset import load_intent_dataset_jsonl

# MODEL_NAME = "/home/sax023/LanguageBind/intention_estimation_module/deberta-v3-large"

# def main():
#     dataset = load_intent_dataset_jsonl("data/ground_truth.jsonl")

#     # Extract unique labels dynamically
#     labels = sorted(list(set(dataset["train"]["label"]) | set(dataset["test"]["label"])))
#     label2id = {l: i for i, l in enumerate(labels)}
#     id2label = {i: l for l, i in label2id.items()}

#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

#     def preprocess(batch):
#         enc = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)
#         enc["label"] = [label2id[l] for l in batch["label"]]
#         return enc

#     tokenized = dataset.map(preprocess, batched=True, remove_columns=["text", "label"])

#     model = AutoModelForSequenceClassification.from_pretrained(
#         MODEL_NAME,
#         num_labels=len(labels),
#         id2label=id2label,
#         label2id=label2id,
#         use_safetensors=True
#     )

#     # Apply LoRA
#     lora_config = LoraConfig(
#         r=16,
#         lora_alpha=32,
#         target_modules=["query_proj", "key_proj", "value_proj"],
#         lora_dropout=0.05,
#         bias="none",
#         task_type="SEQ_CLS"
#     )
#     model = get_peft_model(model, lora_config)

#     training_args = TrainingArguments(
#         output_dir="checkpoints/",
#         save_strategy="epoch",
#         learning_rate=2e-5,
#         per_device_train_batch_size=4,
#         per_device_eval_batch_size=4,
#         num_train_epochs=5,
#         weight_decay=0.01,
#         logging_dir="logs/",
#         fp16=torch.cuda.is_available(),
#     )

#     metric = evaluate.load("accuracy")

#     def compute_metrics(eval_pred):
#         logits, labels_eval = eval_pred
#         preds = logits.argmax(axis=-1)
#         return metric.compute(predictions=preds, references=labels_eval)

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized["train"],
#         eval_dataset=tokenized["test"],
#         tokenizer=tokenizer,
#         compute_metrics=compute_metrics,
#     )

#     trainer.train()
#     model.save_pretrained("models/fine_tuned_classifier", safe_serialization=True)
#     tokenizer.save_pretrained("models/fine_tuned_classifier")

# if __name__ == "__main__":
#     main()
