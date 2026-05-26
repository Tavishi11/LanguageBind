import sys
import types
import torchvision.transforms.functional as _F

_fake = types.ModuleType("torchvision.transforms.functional_tensor")
_fake.__dict__.update({k: getattr(_F, k) for k in dir(_F) if not k.startswith("_")})
sys.modules["torchvision.transforms.functional_tensor"] = _fake

sys.path.insert(0, '/home/sax023/LanguageBind/intention_estimation_module')
sys.path.insert(0, '/home/sax023/LanguageBind')

import os
import time
import json
import torch
import numpy as np
from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer
from models.pipeline import IntentionPipeline
from prompt_rewriting_module.prompt_rewriting import PromptRewritingModule

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

clip_type = {'video': 'LanguageBind_Video_FT'}
model = LanguageBind(clip_type=clip_type, cache_dir='./cache_dir')
model = model.to(device)
model.eval()

tokenizer = LanguageBindImageTokenizer.from_pretrained(
    'LanguageBind/LanguageBind_Image',
    cache_dir='./cache_dir/tokenizer_cache_dir'
)
modality_transform = {c: transform_dict[c](model.modality_config[c]) for c in clip_type.keys()}
pipe = IntentionPipeline()


def get_mini_video_paths(video_uid, clip_start_sec, clip_end_sec):
    base = os.path.expanduser(f"~/video_outputs/{video_uid}/fps_8/mini_videos")
    if not os.path.exists(base):
        print(f"  ✗ No mini-videos for {video_uid}")
        return []
    all_files = sorted(
        [f for f in os.listdir(base) if f.endswith(".mp4")],
        key=lambda x: int(x.replace("mini_video_", "").replace(".mp4", ""))
    )
    return [os.path.join(base, f) for f in all_files]


def get_success_indices(clip_files, clip_start_sec, query_start, query_end, temporal_buffer=2.0):
    """Expands the ground truth window by a small temporal buffer to handle minor frame offsets."""
    success = []
    buffered_start = max(0.0, query_start - temporal_buffer)
    buffered_end = query_end + temporal_buffer

    for idx, f in enumerate(clip_files):
        i = int(os.path.basename(f).replace("mini_video_", "").replace(".mp4", ""))
        seg_start = clip_start_sec + i
        seg_end   = clip_start_sec + i + 1
        if seg_end > buffered_start and seg_start < buffered_end:
            success.append(idx)
    return success


def compute_psucc(similarity_scores, success_indices):
    if not similarity_scores:
        return 0.0
    scores      = np.array(similarity_scores)
    exp_scores  = np.exp(scores - np.max(scores))
    success_set = set(success_indices)
    N_succ = sum(exp_scores[i] for i in range(len(scores)) if i in success_set)
    Z      = exp_scores.sum()
    return float(N_succ / Z) if Z > 0 else 0.0


def run_languagebind_batched(queries, mini_video_paths, batch_size=16):
    lang_input = to_device(
        tokenizer(queries, max_length=77, padding='max_length',
                  truncation=True, return_tensors='pt'), device
    )
    with torch.no_grad():
        lang_embs = model({'language': lang_input})['language']

    all_scores = [[] for _ in range(len(queries))]
    
    for i in range(0, len(mini_video_paths), batch_size):
        batch       = mini_video_paths[i:i + batch_size]
        video_input = to_device(modality_transform['video'](batch), device)
        with torch.no_grad():
            video_emb = model({'video': video_input})['video']
        
        sim_matrix = (video_emb @ lang_embs.T).cpu().tolist()
        
        for q_idx in range(len(queries)):
            all_scores[q_idx].extend([row[q_idx] for row in sim_matrix])
            
    return all_scores


def rewrite_query(query, intent, fallback_rewritten=None):
    # Try Gemini rewrite with fallback chain:
    # 1. Gemini with intent → 2. pre-computed rewritten → 3. original query
    for attempt in range(3):
        try:
            rewriter  = PromptRewritingModule(query, intent)
            rewritten = rewriter.genAIsResponse()
            if rewritten and rewritten.strip():
                return rewritten
            print(f"  ⚠ Empty rewrite (attempt {attempt+1})")
        except Exception as e:
            print(f"  ⚠ Rewrite failed (attempt {attempt+1}): {e}")
            time.sleep(3)
    return None


def safe_atomic_save(data, path):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)

def run_proposed_pipeline(json_path, output_path):
    if os.path.exists(output_path):
        print(f"Found existing evaluation file. Resuming...")
        with open(output_path) as f:
            data = json.load(f)
    else:
        print(f"Creating fresh evaluation state from: {json_path}")
        with open(json_path) as f:
            data = json.load(f)

    total = sum(
        1 for video in data["videos"]
        for clip in video["clips"]
        for ann in clip["annotations"]
        if ann.get("test_set") == "test"
    )
    print(f"Total test annotations: {total}")
    processed_count = 0

    for video in data["videos"]:
        video_uid = video["video_uid"]
        for clip in video["clips"]:
            clip_start_sec = clip["clip_start_sec"]
            clip_end_sec   = clip["clip_end_sec"]
            mini_videos    = None

            for ann in clip["annotations"]:
                if ann.get("test_set") != "test":
                    continue

                processed_count += 1

                query_types = ["query_original", "query_moderate", "query_difficult"]
                needed_types = [qt for qt in query_types if ann.get(f"{qt}_improved_softmax") is None and ann.get(qt)]
                
                if not needed_types:
                    continue

                if mini_videos is None:
                    mini_videos = get_mini_video_paths(video_uid, clip_start_sec, clip_end_sec)

                if not mini_videos:
                    print(f"  [Skipping] No mini-videos for entry {processed_count}")
                    continue

                query_start = ann["query_start"]
                query_end   = ann["query_end"]
                success_idx = get_success_indices(mini_videos, clip_start_sec, query_start, query_end)

                batched_queries = []
                batched_metadata = []

                print(f"\n[{processed_count}/{total}] Processing rewrites...")
                for query_type in needed_types:
                    query = ann.get(query_type)
                    try:
                        intent = json.loads(pipe.process(query))
                        rewritten = rewrite_query(query, intent)

                        if rewritten is None:
                            ann[f"{query_type}_improved_softmax"] = None
                            ann[f"{query_type}_improved_rewritten"] = None
                            continue

                        batched_queries.append(rewritten)
                        batched_metadata.append({
                            'type': query_type,
                            'orig': query,
                            'rewritten': rewritten
                        })
                    except Exception as e:
                        print(f"  ✗ Pre-processing failed for {query_type}: {e}")
                        ann[f"{query_type}_improved_softmax"] = None
                        ann[f"{query_type}_improved_rewritten"] = None

                if batched_queries:
                    try:
                        batched_scores = run_languagebind_batched(batched_queries, mini_videos)

                        for idx, meta in enumerate(batched_metadata):
                            q_type = meta['type']
                            scores = batched_scores[idx]
                            
                            p_succ = compute_psucc(scores, success_idx)
                            best_idx = int(np.argmax(scores))
                            recall_1 = 1.0 if best_idx in success_idx else 0.0

                            ann[f"{q_type}_improved_softmax"] = p_succ
                            ann[f"{q_type}_improved_rewritten"] = meta['rewritten']
                            ann[f"{q_type}_improved_recall_at_1"] = recall_1

                            print(f"  ↳ {q_type} -> psucc: {p_succ:.5f} | recall@1: {recall_1}")

                    except Exception as e:
                        print(f"  ✗ GPU Batched Evaluation failed: {e}")
                        for meta in batched_metadata:
                            ann[f"{meta['type']}_improved_softmax"] = None
                            ann[f"{meta['type']}_improved_rewritten"] = None

                if processed_count % 10 == 0:
                    print(f"\n>>> [Checkpoint] {processed_count}/{total}")
                    safe_atomic_save(data, output_path)

    safe_atomic_save(data, output_path)
    print("\n[Complete]")


if __name__ == "__main__":
    run_proposed_pipeline(
        json_path="intent_dataset/merged_dataset_matched_prompts.json",
        output_path="intent_dataset/merged_dataset_matched_prompts_evaluated_new.json"
    )