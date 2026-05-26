"""
eval_robot_baseline.py — Baseline LanguageBind Evaluation on Robot Study Data

Computes:
  - Cosine similarity: absolute alignment between query and video
  - Softmax score: computed across ALL videos per query (retrieval probability)
    i.e. for each query, how likely is this video vs all others — matches the
    paper's evaluation approach from merged_dataset_matched_prompts.json

Run from: /home/sax023/LanguageBind
Usage:
    python eval_robot_baseline.py
"""

import sys
import os
import torch
import torch.nn.functional as F
import types
import json
import statistics
from collections import defaultdict

# ── Compatibility patch ───────────────────────────────────────────────────────
import torchvision.transforms.functional as _F
_fake = types.ModuleType("torchvision.transforms.functional_tensor")
_fake.__dict__.update({k: getattr(_F, k) for k in dir(_F) if not k.startswith("_")})
sys.modules["torchvision.transforms.functional_tensor"] = _fake

try:
    import torchaudio
    torchaudio.set_audio_backend("soundfile")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
VIDEO_DIR      = "/home/sax023/research_data/eval_extracts"
CACHE_DIR      = "./cache_dir"
QUERY_MAP_PATH = "query_video_mapping.json"


def run_baseline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Video dir: {VIDEO_DIR}\n")

    # ── Load LanguageBind ─────────────────────────────────────────────────────
    print("Loading LanguageBind...")
    clip_type = {"video": "LanguageBind_Video_FT"}
    model = LanguageBind(clip_type=clip_type, cache_dir=CACHE_DIR)
    model = model.to(device)
    model.eval()
    tokenizer = LanguageBindImageTokenizer.from_pretrained(
        "lb203/LanguageBind_Image",
        cache_dir=os.path.join(CACHE_DIR, "tokenizer_cache_dir")
    )
    modality_transform = {c: transform_dict[c](model.modality_config[c]) for c in clip_type}
    print("LanguageBind loaded.\n")

    # ── Load mapping ──────────────────────────────────────────────────────────
    if not os.path.exists(QUERY_MAP_PATH):
        print(f"ERROR: {QUERY_MAP_PATH} not found.")
        return

    with open(QUERY_MAP_PATH, 'r') as f:
        mapping_data = json.load(f)
    video_items = mapping_data.get("videos", {})

    # ── Step 1: Encode ALL videos first ──────────────────────────────────────
    print("Encoding all videos...")
    video_files      = []
    video_embeddings = []

    for video_file, info in video_items.items():
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

    # Stack into matrix [n_videos, emb_dim]
    video_matrix = torch.cat(video_embeddings, dim=0)
    n_videos = len(video_files)
    print(f"\nEncoded {n_videos} videos.\n")

    # ── Step 2: Score each query against ALL videos ───────────────────────────
    # This gives a proper softmax: for a given query, what is the probability
    # that each video is the correct retrieval target?
    # This matches the paper's approach where softmax is computed across segments.

    all_scores = []

    print(f"{'='*70}")
    print("BASELINE EVALUATION — Queries × All Videos (Retrieval Softmax)")
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

        # Encode all queries for this video at once
        lang_inputs = to_device(
            tokenizer(queries, max_length=77, padding="max_length",
                      truncation=True, return_tensors="pt"),
            device
        )
        with torch.no_grad():
            lang_emb = model({"language": lang_inputs})["language"]
        lang_emb_norm = F.normalize(lang_emb, dim=-1)  # [n_queries, emb_dim]

        # Cosine similarity: each query vs ALL videos
        # sim_matrix shape: [n_queries, n_videos]
        sim_matrix = lang_emb_norm @ video_matrix.T

        # Softmax across all videos per query — retrieval probability
        # This answers: "for this query, how likely is each video the correct one?"
        softmax_matrix = torch.softmax(sim_matrix, dim=1)  # softmax over videos

        print(f"\n  {'Query':<52} {'Cosine':>8} {'Softmax':>9} {'Rank':>5}")
        print(f"  {'─'*52} {'─'*8} {'─'*9} {'─'*5}")

        for i, query in enumerate(queries):
            # Cosine to the CORRECT video
            cos  = sim_matrix[i, correct_idx].item()
            # Softmax probability of the CORRECT video
            soft = softmax_matrix[i, correct_idx].item()
            # Rank of correct video among all videos
            rank = (sim_matrix[i].argsort(descending=True) == correct_idx).nonzero(
                as_tuple=True)[0].item() + 1

            q_short = query[:50] + ".." if len(query) > 52 else query
            print(f"  {q_short:<52} {cos:>8.4f} {soft:>9.4f} {rank:>5}")

            all_scores.append({
                "video":    video_file,
                "category": category,
                "query":    query,
                "cosine":   cos,
                "softmax":  soft,
                "rank":     rank,
            })

        avg_cos  = sim_matrix[:, correct_idx].mean().item()
        avg_soft = softmax_matrix[:, correct_idx].mean().item()
        avg_rank = sum(
            (sim_matrix[i].argsort(descending=True) == correct_idx).nonzero(
                as_tuple=True)[0].item() + 1
            for i in range(len(queries))
        ) / len(queries)

        print(f"\n  Avg cosine  : {avg_cos:.4f}")
        print(f"  Avg softmax : {avg_soft:.4f}  (across {n_videos} videos)")
        print(f"  Avg rank    : {avg_rank:.1f} / {n_videos}")

    # ── Overall summary ───────────────────────────────────────────────────────
    if not all_scores:
        print("No scores calculated. Check your video directory paths.")
        return

    all_cos  = [s["cosine"]  for s in all_scores]
    all_soft = [s["softmax"] for s in all_scores]
    all_rank = [s["rank"]    for s in all_scores]

    # Mean Reciprocal Rank — standard retrieval metric
    mrr = statistics.mean(1.0 / r for r in all_rank)
    hit1 = sum(1 for r in all_rank if r == 1) / len(all_rank) * 100
    hit3 = sum(1 for r in all_rank if r <= 3) / len(all_rank) * 100

    print(f"\n\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")
    print(f"  Total queries scored  : {len(all_scores)}")
    print(f"  Videos in pool        : {n_videos}")
    print()
    print(f"  Mean cosine similarity: {statistics.mean(all_cos):.4f}  (std: {statistics.stdev(all_cos):.4f})")
    print(f"  Mean softmax score    : {statistics.mean(all_soft):.4f}  (std: {statistics.stdev(all_soft):.4f})")
    print()
    print(f"  Retrieval Accuracy:")
    print(f"    MRR   (Mean Reciprocal Rank) : {mrr:.4f}")
    print(f"    Hit@1 (correct video = #1)   : {hit1:.1f}%")
    print(f"    Hit@3 (correct video in top3): {hit3:.1f}%")

    print(f"\n  Per video avg cosine & softmax:")
    by_video = defaultdict(lambda: {"cosine": [], "softmax": [], "rank": []})
    for s in all_scores:
        by_video[s["video"]]["cosine"].append(s["cosine"])
        by_video[s["video"]]["softmax"].append(s["softmax"])
        by_video[s["video"]]["rank"].append(s["rank"])

    print(f"  {'Video':<50} {'Cosine':>8} {'Softmax':>9} {'AvgRank':>8}")
    print(f"  {'─'*50} {'─'*8} {'─'*9} {'─'*8}")
    for vid, vals in sorted(by_video.items()):
        mc = statistics.mean(vals["cosine"])
        ms = statistics.mean(vals["softmax"])
        mr = statistics.mean(vals["rank"])
        print(f"  {vid:<50} {mc:>8.4f} {ms:>9.4f} {mr:>8.1f}")

    print(f"{'='*70}\n")

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "summary": {
            "n_queries":   len(all_scores),
            "n_videos":    n_videos,
            "mean_cosine": statistics.mean(all_cos),
            "mean_softmax": statistics.mean(all_soft),
            "mrr":         mrr,
            "hit_at_1":    hit1,
            "hit_at_3":    hit3,
        },
        "results": all_scores,
    }
    with open("eval_robot_baseline_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Results saved to: eval_robot_baseline_results.json")


if __name__ == "__main__":
    run_baseline()