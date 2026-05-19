"""
training/dataset.py
- Consistent label normalization (underscores throughout)
- Merges event_recognition into temporal_localisation
- Augmentation and oversampling for minority classes
- Train / val / test split (70/15/15)
"""

from datasets import Dataset, DatasetDict
import json
import random
from collections import Counter


# ── Label normalization ───────────────────────────────────────────────────────
LABEL_NORMALIZE = {
    "temporal localisation":  "temporal_localisation",
    "temporal_localization":  "temporal_localisation",
    "temporal localisation":  "temporal_localisation",
    "object detection":       "object_detection",
    "event_recognition":      "temporal_localisation",
    "event recognition":      "temporal_localisation",
    "question answering":     "question_answering",
}

VALID_LABELS = {
    "object_detection",
    "temporal_localisation",
    "question_answering",
}


def normalize_label(label: str) -> str:
    label = label.strip().lower()
    return LABEL_NORMALIZE.get(label, label)


# ── Augmentation templates ────────────────────────────────────────────────────
AUGMENTATION_TEMPLATES = {
    "temporal_localisation": [
        "identify the moment when {core}",
        "locate the segment showing {core}",
        "find the time in the video when {core}",
        "show when {core}",
        "at what point does {core}",
    ],
    "question_answering": [
        "can you tell me {core}",
        "what is {core} in this video?",
        "I need to know {core}",
        "could you explain {core}?",
        "help me understand {core}",
    ],
    "object_detection": [
        "can you find {core}",
        "is there {core} in this video?",
        "detect {core}",
        "locate {core} in the scene",
        "show me where {core} is",
    ],
}


def simple_augment(text: str, label: str, n: int = 2) -> list:
    templates = AUGMENTATION_TEMPLATES.get(label, [])
    if not templates:
        return []
    core = text.strip()
    for prefix in ["can you ", "could you ", "show me ", "find ", "detect ",
                   "locate ", "identify ", "what is ", "when does ", "i want to see "]:
        if core.lower().startswith(prefix):
            core = core[len(prefix):]
            break
    chosen = random.sample(templates, min(n, len(templates)))
    return [t.format(core=core) for t in chosen]


# ── Main loader ───────────────────────────────────────────────────────────────
def load_intent_dataset_jsonl(
    path: str = "data/intention_estimation_100.jsonl",
    augment_minority: bool = True,
    oversample: bool = True,
    target_per_class: int = 500,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> DatasetDict:
    random.seed(seed)

    data = []
    with open(path, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            label = normalize_label(entry["task_type"])
            if label not in VALID_LABELS:
                print(f"  [dataset] Skipping unknown label: '{entry['task_type']}'")
                continue
            data.append({"text": entry["prompt"], "label": label})

    print(f"\n[Dataset] Loaded {len(data)} samples after normalization")
    dist = Counter(d["label"] for d in data)
    print(f"[Dataset] Class distribution: {dict(dist)}")

    if augment_minority:
        augmented = []
        majority_count = max(dist.values())
        for item in data:
            if dist[item["label"]] < majority_count * 0.3:
                for v in simple_augment(item["text"], item["label"], n=2):
                    augmented.append({"text": v, "label": item["label"]})
        data.extend(augmented)
        dist = Counter(d["label"] for d in data)
        print(f"[Dataset] After augmentation: {len(data)} samples — {dict(dist)}")

    if oversample:
        by_class = {label: [] for label in VALID_LABELS}
        for item in data:
            by_class[item["label"]].append(item)
            
        balanced = []
        for label, samples in by_class.items():
            if not samples:
                continue
            
            if len(samples) < target_per_class:
                multiplier = (target_per_class // len(samples)) + 1
                samples = (samples * multiplier)[:target_per_class]
            else:
                samples = random.sample(samples, target_per_class)
                
            balanced.extend(samples)
            
        data = balanced
        dist = Counter(d["label"] for d in data)
        print(f"[Dataset] Balanced distribution (target={target_per_class}): {dict(dist)}")

    random.shuffle(data)
    n = len(data)
    n_test = int(n * test_size)
    n_val  = int(n * val_size)
    test_data  = data[:n_test]
    val_data   = data[n_test:n_test + n_val]
    train_data = data[n_test + n_val:]
    print(f"[Dataset] Split — train: {len(train_data)}, val: {len(val_data)}, test: {len(test_data)}")

    return DatasetDict({
        "train":      Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data),
        "test":       Dataset.from_list(test_data),
    })
