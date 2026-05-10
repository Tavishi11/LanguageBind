import sys
import os
import torch
import torch.nn.functional as F
import types
import json  # Fixed: Added missing import
import statistics
from collections import defaultdict

# ── Compatibility patch ───────────────────────────────────────────────────────
import torchvision.transforms.functional as _F
_fake = types.ModuleType("torchvision.transforms.functional_tensor")
_fake.__dict__.update({k: getattr(_F, k) for k in dir(_F) if not k.startswith("_")})
sys.modules["torchvision.transforms.functional_tensor"] = _fake

# Suppress torchaudio warning if backend is not set
try:
    import torchaudio
    torchaudio.set_audio_backend("soundfile")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
VIDEO_DIR  = "/home/sax023/research_data/eval_extracts"
CACHE_DIR  = "./cache_dir"
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

    # ── Load Mapping ──────────────────────────────────────────────────────────
    if not os.path.exists(QUERY_MAP_PATH):
        print(f"ERROR: {QUERY_MAP_PATH} not found.")
        return

    with open(QUERY_MAP_PATH, 'r') as f:
        mapping_data = json.load(f)
    
    video_items = mapping_data.get("videos", {})
    all_scores = []

    print(f"{'='*70}")
    print("BASELINE EVALUATION — Original Queries × Robot Study Videos")
    print(f"{'='*70}")

    for video_file, info in video_items.items():
        video_path = os.path.join(VIDEO_DIR, video_file)
        if not os.path.exists(video_path):
            print(f"\n⚠️  Skipping (not found): {video_file}")
            continue

        category = info["category"]
        gt       = info["ground_truth"]
        queries  = info["queries"]

        print(f"\n{'─'*70}")
        print(f"Video    : {video_file}")
        print(f"Category : {category}")
        print(f"GT Answer: {gt}")
        print(f"Queries  : {len(queries)}")
        print(f"{'─'*70}")

        # Encode video once
        vid_input = {"video": to_device(modality_transform["video"]([video_path]), device)}
        with torch.no_grad():
            video_emb = model(vid_input)["video"]
        video_emb_norm = F.normalize(video_emb, dim=-1)

        # Encode all queries at once
        lang_inputs = to_device(
            tokenizer(queries, max_length=77, padding="max_length",
                      truncation=True, return_tensors="pt"),
            device
        )
        with torch.no_grad():
            lang_emb = model({"language": lang_inputs})["language"]
        lang_emb_norm = F.normalize(lang_emb, dim=-1)

        # Cosine similarity: video vs each query
        # Using a temperature of 10-20 is safer for Softmax than 100
        cosine_scores = (video_emb_norm @ lang_emb_norm.T).squeeze(0)
        softmax_scores = torch.softmax(cosine_scores * 10, dim=0) 

        print(f"\n  {'Query':<55} {'Cosine':>8} {'Softmax':>9}")
        print(f"  {'─'*55} {'─'*8} {'─'*9}")

        for i, query in enumerate(queries):
            cos  = cosine_scores[i].item()
            soft = softmax_scores[i].item()
            q_short = query[:52] + ".." if len(query) > 55 else query
            
            print(f"  {q_short:<55} {cos:>8.4f} {soft:>8.4f}")
            
            all_scores.append({
                "video":    video_file,
                "category": category,
                "query":    query,
                "cosine":   cos,
                "softmax":  soft,
            })

        avg_cos = cosine_scores.mean().item()
        max_cos = cosine_scores.max().item()
        best_q  = queries[cosine_scores.argmax().item()]
        print(f"\n  Avg cosine : {avg_cos:.4f}")
        print(f"  Max cosine : {max_cos:.4f}")
        print(f"  Best query : {best_q}")

    # ── Overall summary ───────────────────────────────────────────────────────
    if not all_scores:
        print("No scores calculated. Check your video directory paths.")
        return

    print(f"\n\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")
    
    all_cos = [s["cosine"] for s in all_scores]
    print(f"  Total queries scored: {len(all_scores)}")
    print(f"  Mean cosine similarity: {statistics.mean(all_cos):.4f}")
    print(f"  Std  cosine similarity: {statistics.stdev(all_cos) if len(all_cos) > 1 else 0:.4f}")
    print(f"  Max  cosine similarity: {max(all_cos):.4f}")
    print(f"  Min  cosine similarity: {min(all_cos):.4f}")

    print(f"\n  Per video avg cosine:")
    by_video = defaultdict(list)
    for s in all_scores:
        by_video[s["video"]].append(s["cosine"])
    
    for vid, scores in sorted(by_video.items()):
        print(f"    {vid:<50} {statistics.mean(scores):.4f}")

    print(f"{'='*70}\n")

    # Save Results
    output_fn = "eval_robot_baseline_results.json"
    with open(output_fn, "w") as f:
        json.dump(all_scores, f, indent=2)
    print(f"Results saved to: {output_fn}")


if __name__ == "__main__":
    run_baseline()