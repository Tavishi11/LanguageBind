"""
tools/compare_vs_baseline.py
Evaluates the improved intention estimation module against the team's baseline
results from the paper (Table 2), using the same 3741-entry test set.

Run from: /home/sax023/LanguageBind/intention_estimation_module

Usage:
    python tools/compare_vs_baseline.py
    python tools/compare_vs_baseline.py --limit 100  # quick test on first 100
"""

import json
import sys
import os
import argparse
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pipeline import IntentionPipeline

# ── Baseline results from paper (Table 2) ─────────────────────────────────────
BASELINE = {
    "task_type": {
        "overall":            63.11,
        "object_detection":   99.96,
        "question_answering":  0.08,
        "temporal_localisation": 0.00,
    },
    "complexity": {
        "overall":   64.77,
        "simple":    64.03,
        "temporal":  97.44,
        "causal":     0.65,
    },
    "temporal_context": {
        "overall":  81.45,
        "none":     78.94,
        "before":   99.12,
        "after":    96.97,
        "during":   68.82,
    },
    "spatial_context": {
        "overall":  57.55,
        "none":     52.32,
        "inside":   74.02,
        "on top of": 93.59,
        "outside":   0.00,
    },
}

FIELDS = ["task_type", "complexity", "temporal_context", "spatial_context"]


def normalize_label(label: str) -> str:
    label = str(label).strip().lower().replace("_", " ")
    mapping = {
        "temporal localisation": "temporal_localisation",
        "temporal localization":  "temporal_localisation",
        "object detection":       "object_detection",
        "question answering":     "question_answering",
        "event recognition":      "temporal_localisation",
        "on top of":              "on_top_of",
        "ontop of":               "on_top_of",
        "in front":               "in_front",
        "left of":                "left_of",
        "right of":               "right_of",
    }
    normalized = label
    return mapping.get(label, normalized)


def evaluate(data_path: str, limit: int = None):
    print(f"\nLoading test data from: {data_path}")
    samples = []
    with open(data_path) as f:
        for line in f:
            samples.append(json.loads(line.strip()))

    if limit:
        samples = samples[:limit]
        print(f"Running on first {limit} samples (use --limit 0 for full set)")

    print(f"Evaluating {len(samples)} samples...\n")
    pipe = IntentionPipeline()

    # Counters
    correct   = {f: defaultdict(int) for f in FIELDS}
    total     = {f: defaultdict(int) for f in FIELDS}
    confidences = []

    for i, sample in enumerate(samples):
        query = sample["prompt"]

        try:
            result_json = pipe.process(query)
            result = json.loads(result_json)
        except Exception as e:
            print(f"  [!] Error on sample {i}: {e}")
            continue

        confidences.append(result.get("confidence", 0))

        for field in FIELDS:
            predicted = normalize_label(result.get(field, "none"))
            expected  = normalize_label(sample.get(field, "none"))
            total[field][expected] += 1
            if predicted == expected:
                correct[field][expected] += 1

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(samples)}...")

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("COMPARISON: YOUR IMPROVED MODULE vs PAPER BASELINE (Table 2)")
    print("=" * 80)

    for field in FIELDS:
        print(f"\n── {field.upper().replace('_',' ')} ──")
        print(f"  {'Class':<25} {'Yours':>8} {'Baseline':>10} {'Delta':>8}")
        print(f"  {'-----':<25} {'-----':>8} {'--------':>10} {'-----':>8}")

        # Overall
        total_correct = sum(correct[field].values())
        total_total   = sum(total[field].values())
        overall_acc   = total_correct / total_total * 100 if total_total > 0 else 0
        baseline_overall = BASELINE.get(field, {}).get("overall", None)
        delta_overall = overall_acc - baseline_overall if baseline_overall else None
        arrow = "▲" if delta_overall and delta_overall > 0 else ("▼" if delta_overall and delta_overall < 0 else " ")
        baseline_str = f"{baseline_overall:.2f}%" if baseline_overall else "N/A"
        delta_str = f"{arrow} {delta_overall:+.2f}%" if delta_overall is not None else "N/A"
        print(f"  {'OVERALL':<25} {overall_acc:>7.2f}% {baseline_str:>10} {delta_str:>8}")

        # Per class
        all_classes = sorted(set(list(total[field].keys()) + list(BASELINE.get(field, {}).keys())) - {"overall"})
        for cls in all_classes:
            t = total[field].get(cls, 0)
            c = correct[field].get(cls, 0)
            acc = c / t * 100 if t > 0 else 0
            baseline_cls = BASELINE.get(field, {}).get(cls, None)
            if baseline_cls is not None:
                delta = acc - baseline_cls
                arrow = "▲" if delta > 0 else ("▼" if delta < 0 else " ")
                delta_str = f"{arrow} {delta:+.2f}%"
                baseline_str = f"{baseline_cls:.2f}%"
            else:
                delta_str = "N/A"
                baseline_str = "N/A"
            print(f"  {cls:<25} {acc:>7.2f}% {baseline_str:>10} {delta_str:>8}  (n={t})")

    avg_conf = sum(confidences) / len(confidences) * 100 if confidences else 0
    print(f"\n── CLASSIFIER CONFIDENCE ──")
    print(f"  Average confidence: {avg_conf:.2f}%")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  type=str, default="../intent_dataset/rewritten_prompts_full_model.json",
                        help="Path to team's test data JSON")
    parser.add_argument("--limit", type=int, default=200,
                        help="Number of samples to evaluate (0 = all 3741, slow)")
    args = parser.parse_args()

    # Convert JSON array to temp JSONL if needed
    data_path = args.data
    if data_path.endswith(".json"):
        tmp_path = "/tmp/eval_data.jsonl"
        with open(data_path) as f:
            data = json.load(f)
        with open(tmp_path, "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")
        data_path = tmp_path

    evaluate(data_path, limit=args.limit if args.limit > 0 else None)


if __name__ == "__main__":
    main()
