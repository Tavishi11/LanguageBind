"""
eval_robot_study.py — End-to-End Pipeline Evaluation on Robot Study Data

Runs the full proposed framework:
  1. Intention Estimation Module (DeBERTa + Mistral)
  2. Prompt Rewriting Module (Gemini)
  3. LanguageBind scoring

Computes the same metrics as eval_robot_baseline.py for direct comparison:
  - Cosine similarity (query embedding vs correct video embedding)
  - Softmax score (across all videos in the pool)
  - MRR, Hit@1, Hit@3 (retrieval accuracy)

Also computes NPIS vs baseline to measure improvement.

Run from: /home/sax023/LanguageBind
Usage:
    python eval_robot_study.py
    python eval_robot_study.py --baseline eval_robot_baseline_results.json
"""

import sys
import os
import torch
import torch.nn.functional as F
import types
import json
import statistics
import time
import argparse
from collections import defaultdict

# ── Compatibility patch ───────────────────────────────────────────────────────
import torchvision.transforms.functional as _F
_fake = types.ModuleType("torchvision.transforms.functional_tensor")
_fake.__dict__.update({k: getattr(_F, k) for k in dir(_F) if not k.startswith("_")})
sys.modules["torchvision.transforms.functional_tensor"] = _fake

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "intention_estimation_module"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "prompt_rewriting_module"))

from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer
from models.pipeline import IntentionPipeline
from prompt_rewriting import PromptRewritingModule

# ── Config ────────────────────────────────────────────────────────────────────
VIDEO_DIR      = "/home/sax023/research_data/eval_extracts"
CACHE_DIR      = "./cache_dir"
QUERY_MAP_PATH = "query_video_mapping.json"
GEMINI_DELAY   = 6  # seconds between Gemini calls to stay under rate limit


def gemini_retry(fn, *args, retries=5, delay=15):
    for i in range(retries):
        try:
            return fn(*args)
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                print(f"    [Gemini rate limit] waiting {delay}s... ({i+1}/{retries})")
                time.sleep(delay)
            else:
                raise
    raise RuntimeError("Gemini rate limit exceeded after retries")


def rewrite_query(raw_query: str, intent: dict) -> str:
    rewriter = PromptRewritingModule(raw_query, intent)
    return rewriter.genAIsResponse().strip()


def npis(orig: float, rewr: float) -> float:
    return (rewr - orig) / abs(orig) if abs(orig) > 1e-8 else 0.0


def run_evaluation(baseline_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── Load LanguageBind ─────────────────────────────────────────────────────
    print("Loading LanguageBind...")
    clip_type = {"video": "LanguageBind_Video_FT"}
    model = LanguageBind(clip_type=clip_type, cache_dir=CACHE_DIR).to(device).eval()
    tokenizer = LanguageBindImageTokenizer.from_pretrained(
        "lb203/LanguageBind_Image",
        cache_dir=os.path.join(CACHE_DIR, "tokenizer_cache_dir")
    )
    modality_transform = {c: transform_dict[c](model.modality_config[c]) for c in clip_type}
    print("LanguageBind loaded.\n")

    # ── Load IntentionPipeline ────────────────────────────────────────────────
    print("Loading IntentionPipeline...")
    orig_dir = os.getcwd()
    os.chdir(os.path.join(SCRIPT_DIR, "intention_estimation_module"))
    intent_pipe = IntentionPipeline()
    os.chdir(orig_dir)
    print("Pipeline loaded.\n")

    # ── Load query mapping ────────────────────────────────────────────────────
    with open(QUERY_MAP_PATH) as f:
        mapping_data = json.load(f)
    video_items = mapping_data.get("videos", {})

    # ── Load baseline results for NPIS comparison ─────────────────────────────
    baseline_lookup = {}
    if baseline_path and os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline_data = json.load(f)
        results_list = baseline_data.get("results", baseline_data)
        if isinstance(results_list, list):
            for r in results_list:
                key = (r["video"], r["query"])
                baseline_lookup[key] = r["cosine"]
        print(f"Loaded {len(baseline_lookup)} baseline scores for NPIS comparison.\n")

    # ── Step 1: Encode ALL videos first ──────────────────────────────────────
    print("Encoding all videos...")
    video_files      = []
    video_embeddings = []

    for video_file in video_items:
        video_path = os.path.join(VIDEO_DIR, video_file)
        if not os.path.exists(video_path):
            print(f"  ⚠️  Not found, skipping: {video_file}")
            continue
        vid_input = {"video": to_device(modality_transform["video"]([video_path]), device)}
        with torch.no_grad():
            emb = model(vid_input)["video"]
        video_files.append(video_file)
        video_embeddings.append(F.normalize(emb, dim=-1))
        print(f"  ✓ {video_file}")

    video_matrix = torch.cat(video_embeddings, dim=0)
    n_videos = len(video_files)
    print(f"\nEncoded {n_videos} videos.\n")

    # ── Step 2: Cache for rewritten prompts ───────────────────────────────────
    # Avoid calling Gemini twice for the same query
    rewrite_cache = {}

    # ── Step 3: Process each query through the full pipeline ─────────────────
    all_results = []

    print(f"{'='*70}")
    print("PROPOSED FRAMEWORK EVALUATION — Intent + Rewrite + LanguageBind")
    print(f"{'='*70}")

    for video_file, info in video_items.items():
        if video_file not in video_files:
            continue

        correct_idx = video_files.index(video_file)
        category    = info["category"]
        gt          = info["ground_truth"]
        queries     = info["queries"]

        print(f"\n{'─'*70}")
        print(f"Video    : {video_file}")
        print(f"Category : {category}")
        print(f"GT Answer: {gt}")
        print(f"Queries  : {len(queries)}")
        print(f"{'─'*70}")

        for raw_query in queries:
            print(f"\n  Query: {raw_query}")

            try:
                # ── Check rewrite cache ───────────────────────────────────────
                if raw_query in rewrite_cache:
                    refined_prompt = rewrite_cache[raw_query]["refined"]
                    intent         = rewrite_cache[raw_query]["intent"]
                    print(f"  [Cache] Reused: {refined_prompt}")
                else:
                    # ── Step 1: Intention Estimation ──────────────────────────
                    intent_raw = intent_pipe.process(raw_query)
                    intent     = json.loads(intent_raw)

                    # ── Step 2: Prompt Rewriting ──────────────────────────────
                    refined_prompt = gemini_retry(rewrite_query, raw_query, intent)
                    rewrite_cache[raw_query] = {
                        "refined": refined_prompt,
                        "intent":  intent,
                    }
                    print(f"  Intent   : {intent.get('task_type','?')} "
                          f"(conf: {intent.get('confidence', 0):.3f})")
                    print(f"  Rewritten: {refined_prompt}")
                    time.sleep(GEMINI_DELAY)

                # ── Step 3: Encode rewritten query ────────────────────────────
                lang_inputs = to_device(
                    tokenizer([refined_prompt], max_length=77, padding="max_length",
                              truncation=True, return_tensors="pt"),
                    device
                )
                with torch.no_grad():
                    lang_emb = model({"language": lang_inputs})["language"]
                lang_emb_norm = F.normalize(lang_emb, dim=-1)  # [1, emb_dim]

                # ── Step 4: Score against ALL videos ─────────────────────────
                # sim_scores shape: [1, n_videos]
                sim_scores = lang_emb_norm @ video_matrix.T
                softmax_scores = torch.softmax(sim_scores, dim=1)

                cos  = sim_scores[0, correct_idx].item()
                soft = softmax_scores[0, correct_idx].item()
                rank = (sim_scores[0].argsort(descending=True) == correct_idx
                        ).nonzero(as_tuple=True)[0].item() + 1

                # ── NPIS vs baseline ──────────────────────────────────────────
                baseline_cos = baseline_lookup.get((video_file, raw_query))
                npis_score   = npis(baseline_cos, cos) if baseline_cos is not None else None

                print(f"  Cosine   : {cos:.4f}  Softmax: {soft:.4f}  "
                      f"Rank: {rank}/{n_videos}"
                      + (f"  NPIS: {npis_score:+.4f}" if npis_score is not None else ""))

                all_results.append({
                    "video":          video_file,
                    "category":       category,
                    "raw_query":      raw_query,
                    "refined_prompt": refined_prompt,
                    "intent":         intent.get("task_type", "?"),
                    "confidence":     intent.get("confidence", 0),
                    "cosine":         cos,
                    "softmax":        soft,
                    "rank":           rank,
                    "baseline_cosine": baseline_cos,
                    "npis":           npis_score,
                })

            except Exception as e:
                print(f"  ERROR: {e}")
                continue

    # ── Summary ───────────────────────────────────────────────────────────────
    if not all_results:
        print("No results computed.")
        return

    all_cos  = [r["cosine"]  for r in all_results]
    all_soft = [r["softmax"] for r in all_results]
    all_rank = [r["rank"]    for r in all_results]

    mrr  = statistics.mean(1.0 / r for r in all_rank)
    hit1 = sum(1 for r in all_rank if r == 1) / len(all_rank) * 100
    hit3 = sum(1 for r in all_rank if r <= 3) / len(all_rank) * 100

    npis_results = [r["npis"] for r in all_results if r["npis"] is not None]
    win_rate     = sum(1 for n in npis_results if n > 0) / len(npis_results) * 100 \
                   if npis_results else None

    print(f"\n\n{'='*70}")
    print("OVERALL SUMMARY — PROPOSED SYSTEM")
    print(f"{'='*70}")
    print(f"  Total queries evaluated : {len(all_results)}")
    print(f"  Videos in pool          : {n_videos}")
    print()
    print(f"  Mean cosine similarity  : {statistics.mean(all_cos):.4f}  "
          f"(std: {statistics.stdev(all_cos):.4f})")
    print(f"  Mean softmax score      : {statistics.mean(all_soft):.4f}  "
          f"(std: {statistics.stdev(all_soft):.4f})")
    print()
    print(f"  Retrieval Accuracy:")
    print(f"    MRR   : {mrr:.4f}")
    print(f"    Hit@1 : {hit1:.1f}%")
    print(f"    Hit@3 : {hit3:.1f}%")

    if npis_results:
        print(f"\n  vs Baseline (NPIS):")
        print(f"    Mean NPIS : {statistics.mean(npis_results):+.4f}")
        print(f"    Win rate  : {win_rate:.1f}% of queries improved")

    # ── Per category breakdown ────────────────────────────────────────────────
    print(f"\n  Per category:")
    by_cat = defaultdict(lambda: {"cosine": [], "rank": [], "npis": []})
    for r in all_results:
        by_cat[r["category"]]["cosine"].append(r["cosine"])
        by_cat[r["category"]]["rank"].append(r["rank"])
        if r["npis"] is not None:
            by_cat[r["category"]]["npis"].append(r["npis"])

    print(f"  {'Category':<40} {'n':>4} {'Cosine':>8} {'AvgRank':>8} {'NPIS':>8}")
    print(f"  {'─'*40} {'─'*4} {'─'*8} {'─'*8} {'─'*8}")
    for cat, vals in sorted(by_cat.items()):
        mc  = statistics.mean(vals["cosine"])
        mr  = statistics.mean(vals["rank"])
        mn  = statistics.mean(vals["npis"]) if vals["npis"] else float("nan")
        cat_short = cat[:38] if len(cat) > 40 else cat
        npis_str = f"{mn:>+8.4f}" if vals["npis"] else "     N/A"
        print(f"  {cat_short:<40} {len(vals['cosine']):>4} {mc:>8.4f} {mr:>8.1f} {npis_str}")

    print(f"{'='*70}\n")

    # ── Save results ──────────────────────────────────────────────────────────
    output = {
        "summary": {
            "n_queries":    len(all_results),
            "n_videos":     n_videos,
            "mean_cosine":  statistics.mean(all_cos),
            "mean_softmax": statistics.mean(all_soft),
            "mrr":          mrr,
            "hit_at_1":     hit1,
            "hit_at_3":     hit3,
            "mean_npis":    statistics.mean(npis_results) if npis_results else None,
            "win_rate":     win_rate,
        },
        "results": all_results,
    }
    with open("eval_robot_study_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Results saved to: eval_robot_study_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, default="eval_robot_baseline_results.json",
                        help="Path to baseline results JSON for NPIS comparison")
    args = parser.parse_args()
    run_evaluation(baseline_path=args.baseline)