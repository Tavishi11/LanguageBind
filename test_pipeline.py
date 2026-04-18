"""
test_pipeline.py
Run from: /home/sax023/LanguageBind

Usage:
    python test_pipeline.py
    python test_pipeline.py --query "Show me when the dog jumps over the fence"
"""

import sys
import json
import argparse

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, "intention_estimation_module")
sys.path.insert(0, "prompt_rewriting_module")

# ── Imports ───────────────────────────────────────────────────────────────────
from models.pipeline import IntentionPipeline
from prompt_rewriting import PromptRewritingModule

# ── Default test queries ──────────────────────────────────────────────────────
DEFAULT_QUERIES = [
    "I jsut bought my new keyboard and I am really struggling to type on this thing... but anyway can you show me when the man leaves his car in this video?",
    "Hey can you find the red cup on the table in this video?",
    "What happens after the woman picks up the phone?",
    "I love dogs! How many dogs are in this video?",
]


def run_pipeline(query: str):
    print("\n" + "=" * 70)
    print(f"INPUT QUERY:\n  {query}")
    print("=" * 70)

    # ── Step 1: Intention Estimation ──────────────────────────────────────────
    print("\n[Step 1] Running Intention Estimation Module...")
    pipe = IntentionPipeline()
    intent_json = pipe.process(query)
    intent = json.loads(intent_json)

    print(f"\n  Classified Intent:")
    for k, v in intent.items():
        if k != "prompt":
            print(f"    {k}: {v}")

    # ── Step 2: Prompt Rewriting ──────────────────────────────────────────────
    print("\n[Step 2] Running Prompt Rewriting Module...")

    # Map intent fields to what PromptRewritingModule expects
    user_intent = {
        "task_type":        intent.get("task_type", "None"),
        "output_modality":  intent.get("output_modality", "Video"),
        "complexity":       intent.get("complexity", "None"),
        "temporal_context": intent.get("temporal_context", "None"),
        "spatial_context":  intent.get("spatial_context", "None"),
    }

    rewriter = PromptRewritingModule(query, user_intent)
    rewritten = rewriter.genAIsResponse()

    print(f"\n  Rewritten Prompt:\n    {rewritten.strip()}")

    # ── Final Summary ─────────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("PIPELINE SUMMARY")
    print("-" * 70)
    print(f"  Original : {query}")
    print(f"  Intent   : {user_intent}")
    print(f"  Rewritten: {rewritten.strip()}")
    print("=" * 70 + "\n")

    return {
        "original": query,
        "intent": intent,
        "rewritten": rewritten.strip()
    }


def main():
    parser = argparse.ArgumentParser(description="Test intention estimation + prompt rewriting pipeline")
    parser.add_argument("--query", type=str, default=None, help="Single query to test")
    parser.add_argument("--all", action="store_true", help="Run all default test queries")
    args = parser.parse_args()

    if args.query:
        run_pipeline(args.query)
    elif args.all:
        for q in DEFAULT_QUERIES:
            run_pipeline(q)
    else:
        # Interactive mode
        print("Pipeline Test — Intention Estimation + Prompt Rewriting")
        print("Type 'quit' to exit, 'all' to run default test queries\n")
        while True:
            query = input("Enter query: ").strip()
            if query.lower() == "quit":
                break
            elif query.lower() == "all":
                for q in DEFAULT_QUERIES:
                    run_pipeline(q)
            elif query:
                run_pipeline(query)


if __name__ == "__main__":
    main()
