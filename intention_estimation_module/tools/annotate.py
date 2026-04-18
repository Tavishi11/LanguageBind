"""
tools/annotate.py
Rule-based annotation tool for intention estimation dataset.

Automatically labels prompts using keyword rules, then lets you
review/correct borderline cases interactively.

Usage:
    # Annotate a file of raw prompts (one per line)
    python tools/annotate.py --input raw_prompts.txt --output data/new_annotations.jsonl

    # Review and correct auto-annotations interactively
    python tools/annotate.py --input raw_prompts.txt --output data/new_annotations.jsonl --review

    # Merge new annotations into ground truth
    python tools/annotate.py --merge data/new_annotations.jsonl --ground-truth data/ground_truth.jsonl
"""

import json
import re
import argparse
from collections import Counter


# ── Rule definitions ──────────────────────────────────────────────────────────

TEMPORAL_LOCALISATION_RULES = [
    # Strong temporal keywords
    r"\bwhen\b",
    r"\bwhenever\b",
    r"\bat what (point|moment|time)\b",
    r"\bmoment\b",
    r"\binstant\b",
    r"\bhow long\b",
    r"\bstart(s)? (of|when)\b",
    r"\bend(s)? (of|when)\b",
    r"\bbefore\b",
    r"\bafter\b",
    r"\bduring\b",
    r"\bwhile\b",
    r"\buntil\b",
    r"\bonce\b",
    r"\bfirst.*then\b",
    r"\bsequen\w+\b",
    r"\btimeline\b",
    r"\btemporal\b",
    r"\blocali[sz]e\b",
    r"\blocate (when|the moment|the time)\b",
    r"\bshow (me )?(when|the moment)\b",
    r"\bfind (when|the moment|the time)\b",
    r"\bidentify (when|the moment)\b",
]

OBJECT_DETECTION_RULES = [
    r"\bdetect\b",
    r"\bfind (the|a|an)\b",
    r"\blocate (the|a|an)\b",
    r"\bwhere is\b",
    r"\bwhere are\b",
    r"\bshow (me )?(the|a|an)\b",
    r"\bspot\b",
    r"\bidentify (the|a|an)\b",
    r"\bis there (a|an|the)\b",
    r"\bare there\b",
    r"\bcan you (find|see|spot)\b",
    r"\blook for\b",
    r"\bsearch for\b",
    r"\bpoint out\b",
    r"\bhighlight\b",
    r"\bmark\b",
]

QUESTION_ANSWERING_RULES = [
    r"\bhow many\b",
    r"\bhow much\b",
    r"\bwhat (is|are|was|were|colour|color|type|kind)\b",
    r"\bwhat happened\b",
    r"\bwhat happens\b",
    r"\bwhy\b",
    r"\bwhich\b",
    r"\bwho\b",
    r"\bwhose\b",
    r"\bcount\b",
    r"\btell me\b",
    r"\bexplain\b",
    r"\bdescribe\b",
    r"\bwhat caused\b",
    r"\bwhat led\b",
    r"\breason\b",
    r"\bcause\b",
]

COMPLEXITY_RULES = {
    "causal": [
        r"\bwhy\b", r"\bbecause\b", r"\bcause[sd]?\b", r"\bdue to\b",
        r"\bled to\b", r"\bresult(ed)? in\b", r"\btrigger\b", r"\beffect\b",
        r"\bconsequen\w+\b", r"\bwhat caused\b", r"\bwhat made\b",
    ],
    "temporal": [
        r"\bbefore\b", r"\bafter\b", r"\bduring\b", r"\bwhile\b",
        r"\bfirst.*then\b", r"\bsequen\w+\b", r"\bonce\b", r"\buntil\b",
        r"\bthen\b", r"\bprevious\b", r"\bsubsequent\b",
    ],
}

TEMPORAL_CONTEXT_RULES = {
    "before": [r"\bbefore\b", r"\bprior to\b", r"\bpreceding\b"],
    "after":  [r"\bafter\b",  r"\bfollowing\b", r"\bsubsequent\b", r"\bonce.*done\b"],
    "during": [r"\bduring\b", r"\bwhile\b", r"\bin the middle of\b", r"\bmeanwhile\b"],
}

SPATIAL_CONTEXT_RULES = {
    "inside":    [r"\binside\b",    r"\bwithin\b",    r"\bin the\b"],
    "in front":  [r"\bin front\b",  r"\bfacing\b",    r"\bin front of\b"],
    "left of":   [r"\bleft of\b",   r"\bto the left\b"],
    "right of":  [r"\bright of\b",  r"\bto the right\b"],
    "behind":    [r"\bbehind\b",    r"\bat the back\b", r"\bin the back\b"],
    "on top of": [r"\bon top of\b", r"\babove\b",     r"\bon the\b"],
    "under":     [r"\bunder\b",     r"\bbeneath\b",   r"\bbelow\b", r"\bunderneath\b"],
}


# ── Rule engine ───────────────────────────────────────────────────────────────

def match_rules(text: str, rules: list) -> int:
    """Count how many rules match the text."""
    text = text.lower()
    return sum(1 for r in rules if re.search(r, text))


def annotate_prompt(prompt: str) -> dict:
    """
    Auto-annotate a prompt using keyword rules.
    Returns annotation dict with confidence scores.
    """
    text = prompt.lower()

    # ── Task type ─────────────────────────────────────────────────────────────
    temporal_score = match_rules(text, TEMPORAL_LOCALISATION_RULES)
    object_score   = match_rules(text, OBJECT_DETECTION_RULES)
    qa_score       = match_rules(text, QUESTION_ANSWERING_RULES)

    scores = {
        "temporal_localisation": temporal_score,
        "object_detection":      object_score,
        "question_answering":    qa_score,
    }
    total = sum(scores.values()) or 1
    task_type = max(scores, key=scores.get)

    # Confidence: how dominant is the winner
    confidence = scores[task_type] / total if total > 0 else 0.0
    ambiguous  = sorted(scores.values(), reverse=True)
    is_ambiguous = len(ambiguous) > 1 and ambiguous[0] - ambiguous[1] <= 1

    # ── Complexity ────────────────────────────────────────────────────────────
    causal_score   = match_rules(text, COMPLEXITY_RULES["causal"])
    temporal_c_score = match_rules(text, COMPLEXITY_RULES["temporal"])

    if causal_score > 0:
        complexity = "causal"
    elif temporal_c_score > 0:
        complexity = "temporal"
    else:
        complexity = "simple"

    # ── Temporal context ──────────────────────────────────────────────────────
    temporal_context = "none"
    for ctx, rules in TEMPORAL_CONTEXT_RULES.items():
        if match_rules(text, rules) > 0:
            temporal_context = ctx
            break

    # ── Spatial context ───────────────────────────────────────────────────────
    spatial_context = "none"
    for ctx, rules in SPATIAL_CONTEXT_RULES.items():
        if match_rules(text, rules) > 0:
            spatial_context = ctx
            break

    return {
        "prompt":           prompt.strip(),
        "task_type":        task_type,
        "complexity":       complexity,
        "temporal_context": temporal_context,
        "spatial_context":  spatial_context,
        "_confidence":      round(confidence, 2),   # internal, not saved to jsonl
        "_ambiguous":       is_ambiguous,            # internal, flags for review
        "_scores":          scores,                  # internal, for review display
    }


# ── Interactive review ────────────────────────────────────────────────────────

VALID_TASK_TYPES = ["temporal_localisation", "object_detection", "question_answering"]
VALID_COMPLEXITY = ["simple", "causal", "temporal"]
VALID_TEMPORAL   = ["none", "before", "after", "during"]
VALID_SPATIAL    = ["none", "inside", "in front", "left of", "right of", "behind", "on top of", "under"]

SHORTCUTS = {
    "t": "temporal_localisation",
    "o": "object_detection",
    "q": "question_answering",
    "s": "simple",
    "c": "causal",
    "te": "temporal",
}


def review_annotation(annotation: dict, idx: int, total: int) -> dict:
    """Interactively review and correct a single annotation."""
    print(f"\n{'─'*60}")
    print(f"[{idx}/{total}] {'⚠️  AMBIGUOUS' if annotation['_ambiguous'] else '✓  AUTO'}")
    print(f"Prompt    : {annotation['prompt']}")
    print(f"Scores    : {annotation['_scores']}")
    print(f"task_type : {annotation['task_type']}  (confidence={annotation['_confidence']})")
    print(f"complexity: {annotation['complexity']}")
    print(f"temporal  : {annotation['temporal_context']}")
    print(f"spatial   : {annotation['spatial_context']}")
    print()
    print("Press ENTER to accept, or type corrections:")
    print("  task_type  → t=temporal_localisation, o=object_detection, q=question_answering")
    print("  complexity → s=simple, c=causal, te=temporal")
    print("  's' to skip, 'd' to discard")

    response = input("> ").strip().lower()

    if response == "s":
        return annotation   # keep as-is
    if response == "d":
        return None         # discard
    if response == "":
        return annotation   # accept

    # Parse corrections
    parts = response.split()
    for part in parts:
        if part in SHORTCUTS:
            val = SHORTCUTS[part]
            if val in VALID_TASK_TYPES:
                annotation["task_type"] = val
            elif val in VALID_COMPLEXITY:
                annotation["complexity"] = val
        elif part in VALID_TASK_TYPES:
            annotation["task_type"] = part
        elif part in VALID_COMPLEXITY:
            annotation["complexity"] = part
        elif part in VALID_TEMPORAL:
            annotation["temporal_context"] = part
        elif part in VALID_SPATIAL:
            annotation["spatial_context"] = part

    return annotation


def clean_annotation(ann: dict) -> dict:
    """Remove internal fields before saving."""
    return {k: v for k, v in ann.items() if not k.startswith("_")}


# ── Main ──────────────────────────────────────────────────────────────────────

def run_annotate(input_path: str, output_path: str, review: bool = False):
    with open(input_path) as f:
        prompts = [l.strip() for l in f if l.strip()]

    print(f"\nAnnotating {len(prompts)} prompts...")
    annotations = [annotate_prompt(p) for p in prompts]

    # Stats
    dist = Counter(a["task_type"] for a in annotations)
    ambiguous_count = sum(1 for a in annotations if a["_ambiguous"])
    print(f"Auto-annotation complete:")
    print(f"  Distribution : {dict(dist)}")
    print(f"  Ambiguous    : {ambiguous_count}/{len(annotations)} flagged for review")

    if review:
        print(f"\nReview mode — reviewing all {len(annotations)} annotations")
        print("(Only ambiguous ones are marked ⚠️)")
        reviewed = []
        for i, ann in enumerate(annotations):
            result = review_annotation(ann, i + 1, len(annotations))
            if result is not None:
                reviewed.append(result)
        annotations = reviewed
        print(f"\n{len(annotations)} annotations kept after review.")
    else:
        print(f"\nSkipping review. Use --review to interactively correct annotations.")

    # Save
    with open(output_path, "w") as f:
        for ann in annotations:
            f.write(json.dumps(clean_annotation(ann)) + "\n")

    print(f"Saved {len(annotations)} annotations to {output_path}")
    dist = Counter(a["task_type"] for a in annotations)
    print(f"Final distribution: {dict(dist)}")


def run_merge(new_path: str, ground_truth_path: str):
    """Merge new annotations into ground truth, deduplicating by prompt."""
    existing = {}
    with open(ground_truth_path) as f:
        for line in f:
            entry = json.loads(line)
            existing[entry["prompt"].strip().lower()] = entry

    added = 0
    with open(new_path) as f:
        for line in f:
            entry = json.loads(line)
            key = entry["prompt"].strip().lower()
            if key not in existing:
                existing[key] = entry
                added += 1

    with open(ground_truth_path, "w") as f:
        for entry in existing.values():
            f.write(json.dumps(entry) + "\n")

    print(f"Merged: {added} new entries added. Total: {len(existing)} samples.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",        type=str, help="Input file of raw prompts (one per line)")
    parser.add_argument("--output",       type=str, help="Output JSONL file for annotations")
    parser.add_argument("--review",       action="store_true", help="Interactively review annotations")
    parser.add_argument("--merge",        type=str, help="New annotations JSONL to merge")
    parser.add_argument("--ground-truth", type=str, default="data/ground_truth.jsonl")
    args = parser.parse_args()

    if args.merge:
        run_merge(args.merge, args.ground_truth)
    elif args.input and args.output:
        run_annotate(args.input, args.output, review=args.review)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
