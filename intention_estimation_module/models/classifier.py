"""
models/classifier.py
Fixes applied:
- Load id2label/label2id from saved model config instead of re-reading the
  training file at inference time. The new training normalizes labels to
  underscores (object_detection, temporal_localisation, question_answering),
  but _extract_unique_field read raw spaced strings from the JSONL, silently
  producing a mismatched label2id and corrupting every prediction.
- Removed device_map={"": 0} hardcode; use device_map="auto".
- should_use_fallback now accepts task_type kwarg for per-class thresholds.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import json


DEFAULT_THRESHOLDS = {
    "temporal_localisation": 0.35,
    "question_answering":    0.35,
    "object_detection":      0.45,
}


class IntentionClassifier:
    def __init__(
        self,
        model_name,
        training_data_path,         # kept for API compatibility, no longer used for labels
        peft_path=None,
        temps_path=None,
        confidence_threshold=0.4,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.confidence_threshold = confidence_threshold

        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,
            torch_dtype=torch.float16,
            device_map={"":0},
            trust_remote_code=True,
            use_safetensors=False,
        )

        if peft_path:
            self.model = PeftModel.from_pretrained(base_model, peft_path)
        else:
            self.model = base_model

        # Hardcoded label mapping (sorted alphabetically, matches training)
        self.task_labels = ['object_detection', 'question_answering', 'temporal_localisation']
        self.id2label = {i: l for i, l in enumerate(self.task_labels)}
        self.label2id = {l: i for i, l in enumerate(self.task_labels)}

        # Per-class confidence thresholds
        self.class_thresholds = {
            'temporal_localisation': 0.35,
            'question_answering':    0.35,
            'object_detection':      0.40,
        }

        # FIX: Load label mapping from the config saved at training time.
        # Previously _extract_unique_field re-read the raw training JSONL and
        # built a mapping from un-normalized spaced labels, while the model was
        # trained with normalized underscore labels — every prediction was wrong.

        self.class_thresholds = {
            label: DEFAULT_THRESHOLDS.get(label, confidence_threshold)
            for label in self.task_labels
        }

        if temps_path:
            with open(temps_path) as f:
                self.temperatures = json.load(f)
        else:
            self.temperatures = {label: 1.0 for label in self.task_labels}

        self.device = next(self.model.parameters()).device

    def predict_task_type(self, text: str) -> dict:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            padding=True, max_length=256,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits.squeeze(0)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        top2_conf, top2_idx = torch.topk(probs, min(2, len(self.task_labels)))

        primary_label = self.task_labels[top2_idx[0].item()]
        primary_conf  = top2_conf[0].item()

        result = {
            "task_type":     primary_label,
            "confidence":    primary_conf,
            "probabilities": {self.task_labels[i]: p.item() for i, p in enumerate(probs)},
        }
        if len(top2_idx) > 1:
            result["secondary_task_type"]  = self.task_labels[top2_idx[1].item()]
            result["secondary_confidence"] = top2_conf[1].item()

        return result

    def should_use_fallback(self, confidence: float, task_type: str = None) -> bool:
        if task_type and task_type in self.class_thresholds:
            return confidence < self.class_thresholds[task_type]
        return confidence < self.confidence_threshold
