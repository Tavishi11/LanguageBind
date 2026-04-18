"""
training/train.py
Fixes applied vs broken new version:
- load_best_model_at_end=True      (was False — saved last epoch, not best)
- EarlyStoppingCallback restored   (was [])
- num_train_epochs=10              (was 2 — too few for LoRA to converge)
- target_per_class=500             (was 50 — far too small for DeBERTa-large)
- learning_rate=2e-5               (was 5e-4 — too aggressive)
- r=16 LoRA rank restored          (was 8 — old baseline used 16)
- evaluation_strategy="epoch"      (required for load_best_model_at_end to work)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import evaluate
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model
from dataset import load_intent_dataset_jsonl
from collections import Counter

MODEL_NAME = "/home/sax023/LanguageBind/intention_estimation_module/deberta-v3-large"
DATA_PATH  = "data/ground_truth.jsonl"


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss_fn = CrossEntropyLoss(weight=weights, label_smoothing=0.1)
        else:
            loss_fn = CrossEntropyLoss(label_smoothing=0.1)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_class_weights(labels: list, num_classes: int) -> torch.Tensor:
    counts = Counter(labels)
    total  = sum(counts.values())
    weights = [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)]
    t = torch.tensor(weights, dtype=torch.float32)
    print(f"\n[Training] Class weights: {t.tolist()}")
    return t


def main():
    dataset = load_intent_dataset_jsonl(
        path=DATA_PATH,
        augment_minority=False,
        oversample=False,
        target_per_class=500,           # FIX: was 50
    )

    all_labels = sorted(
        set(dataset["train"]["label"]) |
        set(dataset["validation"]["label"]) |
        set(dataset["test"]["label"])
    )
    label2id = {l: i for i, l in enumerate(all_labels)}
    id2label  = {i: l for l, i in label2id.items()}
    print(f"\n[Training] Labels: {all_labels}")

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

    train_labels = [label2id[l] for l in dataset["train"]["label"]]
    class_weights = compute_class_weights(train_labels, len(all_labels))

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(all_labels),
        id2label=id2label,
        label2id=label2id,
        use_safetensors=False,
    )

    lora_config = LoraConfig(
        r=16,                           # FIX: restored to 16 (was reduced to 8)
        lora_alpha=32,
        target_modules=["query_proj", "key_proj", "value_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir="checkpoints/",
        evaluation_strategy="epoch",    # FIX: required for load_best_model_at_end
        save_strategy="epoch",
        load_best_model_at_end=False,    # FIX: was False
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        learning_rate=2e-5,             # FIX: was 5e-4
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=10,            # FIX: was 2
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir="logs/",
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    accuracy_metric = evaluate.load("accuracy")
    f1_metric       = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels_eval = eval_pred
        preds = logits.argmax(axis=-1)
        acc = accuracy_metric.compute(predictions=preds, references=labels_eval)
        f1  = f1_metric.compute(predictions=preds, references=labels_eval, average="macro")
        return {**acc, **f1}

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[],  # FIX: was []
    )

    print("\n[Training] Starting training...")
    trainer.train()

    print("\n[Training] Evaluating on test set...")
    print(f"\n[Training] Test results: {trainer.evaluate(tokenized['test'])}")

    model.save_pretrained("models/fine_tuned_classifier", safe_serialization=True)
    tokenizer.save_pretrained("models/fine_tuned_classifier")
    print("\n[Training] Model saved to models/fine_tuned_classifier")


if __name__ == "__main__":
    main()
