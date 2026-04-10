import torchvision.transforms.functional as _F
import sys, types
_fake = types.ModuleType("torchvision.transforms.functional_tensor")
_fake.__dict__.update({k: getattr(_F, k) for k in dir(_F) if not k.startswith("_")})
sys.modules["torchvision.transforms.functional_tensor"] = _fake

"""
test_pipeline.py
Runs the full pipeline: Intention Estimation → Prompt Rewriting → LanguageBind similarity scoring.
Compares original vs rewritten prompt similarity scores against provided videos.

Run from: /home/sax023/LanguageBind/intention_estimation_module

Usage:
    python test_pipeline.py --query "Show me when the dog jumps over the fence" --video ../assets/video/0.mp4
    python test_pipeline.py --all
    python test_pipeline.py  # interactive mode
"""

import os
import sys
import json
import argparse

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
LANGUAGEBIND_DIR = os.path.dirname(SCRIPT_DIR)

sys.path.insert(0, SCRIPT_DIR)                                                 # intention_estimation_module/
sys.path.insert(0, os.path.join(LANGUAGEBIND_DIR, "prompt_rewriting_module"))  # prompt_rewriting_module/
sys.path.insert(0, LANGUAGEBIND_DIR)                                           # LanguageBind/ root

# ── Imports ───────────────────────────────────────────────────────────────────
import torch
from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer
from models.pipeline import IntentionPipeline
from prompt_rewriting import PromptRewritingModule

# ── LanguageBind setup (loaded once) ─────────────────────────────────────────
print("Loading LanguageBind model...")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIP_TYPE  = {"video": "LanguageBind_Video_FT"}
LB_MODEL   = LanguageBind(clip_type=CLIP_TYPE, cache_dir=os.path.join(LANGUAGEBIND_DIR, "cache_dir"))
LB_MODEL   = LB_MODEL.to(DEVICE)
LB_MODEL.eval()
TOKENIZER  = LanguageBindImageTokenizer.from_pretrained(
    "lb203/LanguageBind_Image",
    cache_dir=os.path.join(LANGUAGEBIND_DIR, "cache_dir/tokenizer_cache_dir")
)
MODALITY_TRANSFORM = {c: transform_dict[c](LB_MODEL.modality_config[c]) for c in CLIP_TYPE.keys()}
print("LanguageBind model loaded.\n")

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_QUERIES = [
    "I just bought my new keyboard and I am really struggling to type on this thing... but anyway can you show me when the man leaves his car in this video?",
    "Hey can you find the red cup on the table in this video?",
    "What happens after the woman picks up the phone?",
    "I love dogs! How many dogs are in this video?",
]

DEFAULT_VIDEOS = [
    os.path.join(LANGUAGEBIND_DIR, "assets/video/0.mp4"),
    os.path.join(LANGUAGEBIND_DIR, "assets/video/1.mp4"),
]


def score_prompt(prompt: str, video_paths: list) -> dict:
    """Score a prompt against videos using LanguageBind softmax similarity."""
    inputs = {
        "video": to_device(MODALITY_TRANSFORM["video"](video_paths), DEVICE),
    }
    inputs["language"] = to_device(
        TOKENIZER([prompt], max_length=77, padding="max_length", truncation=True, return_tensors="pt"),
        DEVICE
    )
    with torch.no_grad():
        embeddings = LB_MODEL(inputs)

    scores = torch.softmax(embeddings["video"] @ embeddings["language"].T, dim=0)
    return {os.path.basename(v): round(scores[i].item(), 4) for i, v in enumerate(video_paths)}


import time

def run_pipeline(query: str, video_paths: list):
    print("\n" + "=" * 70)
    print(f"INPUT QUERY:\n  {query}")
    print(f"VIDEOS: {[os.path.basename(v) for v in video_paths]}")
    print("=" * 70)

    # ── Step 1: Intention Estimation ──────────────────────────────────────────
    print("\n[Step 1] Running Intention Estimation Module...")
    pipe = IntentionPipeline()
    intent_json = pipe.process(query)
    intent = json.loads(intent_json)

    print("\n  Classified Intent:")
    for k, v in intent.items():
        if k != "prompt":
            print(f"    {k}: {v}")

    # ── Step 2: Prompt Rewriting ──────────────────────────────────────────────
    print("\n[Step 2] Running Prompt Rewriting Module...")
    user_intent = {
        "task_type":        intent.get("task_type", "None"),
        "output_modality":  intent.get("output_modality", "Video"),
        "complexity":       intent.get("complexity", "None"),
        "temporal_context": intent.get("temporal_context", "None"),
        "spatial_context":  intent.get("spatial_context", "None"),
    }
    rewriter  = PromptRewritingModule(query, user_intent)
    rewritten = rewriter.genAIsResponse().strip()
    print(f"\n  Rewritten Prompt: {rewritten}")

    # ── Step 3: LanguageBind Similarity Scoring ───────────────────────────────
    print("\n[Step 3] Scoring with LanguageBind...")
    original_scores  = score_prompt(query,     video_paths)
    rewritten_scores = score_prompt(rewritten, video_paths)

    # ── Results table ─────────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("SIMILARITY SCORES  (softmax — higher = more relevant video)")
    print("-" * 70)
    print(f"  {'Video':<20} {'Original':>10} {'Rewritten':>10} {'Delta':>10}")
    print(f"  {'-----':<20} {'--------':>10} {'---------':>10} {'-----':>10}")

    for video in original_scores:
        orig  = original_scores[video]
        rew   = rewritten_scores[video]
        delta = rew - orig
        arrow = "▲" if delta > 0.0001 else ("▼" if delta < -0.0001 else "─")
        print(f"  {video:<20} {orig:>10.4f} {rew:>10.4f} {arrow} {delta:>+.4f}")

    avg_orig  = sum(original_scores.values())  / len(original_scores)
    avg_rew   = sum(rewritten_scores.values()) / len(rewritten_scores)
    avg_delta = avg_rew - avg_orig

    print(f"  {'':─<64}")
    print(f"  {'AVERAGE':<20} {avg_orig:>10.4f} {avg_rew:>10.4f}   {avg_delta:>+.4f}")
    print("-" * 70)
    print(f"\n  Classifier confidence : {intent.get('confidence', 0):.3f}")
    print(f"  Original  : {query}")
    print(f"  Rewritten : {rewritten}")
    improvement = "IMPROVED ✓" if avg_delta > 0 else ("DEGRADED ✗" if avg_delta < 0 else "NO CHANGE")
    print(f"  Result    : {improvement}")
    print("=" * 70 + "\n")

    return {
        "original":         query,
        "rewritten":        rewritten,
        "intent":           intent,
        "original_scores":  original_scores,
        "rewritten_scores": rewritten_scores,
        "avg_delta":        avg_delta,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str,   default=None, help="Single query to test")
    parser.add_argument("--video", type=str,   default=None, nargs="+", help="Video file path(s)")
    parser.add_argument("--all",   action="store_true",      help="Run all default test queries")
    args = parser.parse_args()

    video_paths = args.video if args.video else DEFAULT_VIDEOS
    video_paths = [v for v in video_paths if os.path.exists(v)]
    if not video_paths:
        print("ERROR: No valid video files found. Check --video paths or assets/video/ folder.")
        sys.exit(1)

    if args.query:
        run_pipeline(args.query, video_paths)
    elif args.all:
        for q in DEFAULT_QUERIES:
            run_pipeline(q, video_paths)
    else:
        print("Pipeline Test — Intention Estimation + Prompt Rewriting + LanguageBind")
        print("Type 'quit' to exit, 'all' to run all default queries\n")
        while True:
            query = input("Enter query: ").strip()
            if query.lower() == "quit":
                break
            elif query.lower() == "all":
                for q in DEFAULT_QUERIES:
                    run_pipeline(q, video_paths)
            elif query:
                run_pipeline(query, video_paths)


if __name__ == "__main__":
    main()
