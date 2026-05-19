import json
from collections import defaultdict
from models.pipeline import IntentionPipeline

def generate_intent_stats(test_data_path):
    pipe = IntentionPipeline()
    
    with open(test_data_path, 'r') as f:
        data = json.load(f)

    # intent_test_dataset.json is already filtered to test set
    test_samples = data

    stats = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
    dimensions = ["task_type", "complexity", "temporal_context", "spatial_context"]

    print(f"Processing {len(test_samples)} samples...")

    for i, sample in enumerate(test_samples):
        if i % 100 == 0:
            print(f"  [{i}/{len(test_samples)}]")

        raw_query = sample["prompt"]
        prediction_json = pipe.process(raw_query)
        prediction = json.loads(prediction_json)

        for dim in dimensions:
            gt_label   = str(sample.get(dim, "")).lower().strip()
            pred_label = str(prediction.get(dim, "")).lower().strip()

            stats[dim][gt_label]["total"] += 1
            if gt_label == pred_label:
                stats[dim][gt_label]["correct"] += 1

    print(f"\n{'Dimension':<20} | {'Class':<20} | {'Correct':<8} | {'Total':<8} | {'Accuracy %'}")
    print("-" * 85)
    for dim, classes in stats.items():
        dim_correct = 0
        dim_total   = 0
        for cls_name, counts in classes.items():
            acc = (counts["correct"] / counts["total"]) * 100 if counts["total"] > 0 else 0
            print(f"{dim:<20} | {cls_name:<20} | {counts['correct']:<8} | {counts['total']:<8} | {acc:.2f}%")
            dim_correct += counts["correct"]
            dim_total   += counts["total"]
        overall_acc = (dim_correct / dim_total) * 100 if dim_total > 0 else 0
        print(f"--- OVERALL {dim.upper()}: {dim_correct}/{dim_total} ({overall_acc:.2f}%) ---\n")

if __name__ == "__main__":
    generate_intent_stats("../intent_dataset/intent_test_dataset.json")