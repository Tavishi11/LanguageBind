import sys
import os
import torch
import torch.nn.functional as F
import types
import json
import statistics
import time
from collections import defaultdict

# ── LanguageBind Compatibility Patch ──────────────────────────────────────────
import torchvision.transforms.functional as _F
_fake = types.ModuleType("torchvision.transforms.functional_tensor")
_fake.__dict__.update({k: getattr(_F, k) for k in dir(_F) if not k.startswith("_")})
sys.modules["torchvision.transforms.functional_tensor"] = _fake

# ── Dynamic Path Setup for Modules ───────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "intention_estimation_module"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "prompt_rewriting_module"))

from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer
from models.pipeline import IntentionPipeline
from prompt_rewriting import PromptRewritingModule

# ── Configuration ─────────────────────────────────────────────────────────────
VIDEO_DIR  = "/home/sax023/research_data/eval_extracts"
CACHE_DIR  = "./cache_dir"
QUERY_MAP  = "query_video_mapping.json"
# Stay under Gemini RPM limits (Approx 10-12 requests per minute)
GEMINI_DELAY = 6.5 

def run_proposed_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize Pipeline Components
    print("Initializing Intention Estimation & LanguageBind...")
    model = LanguageBind(clip_type={"video": "LanguageBind_Video_FT"}, cache_dir=CACHE_DIR).to(device).eval()
    tokenizer = LanguageBindImageTokenizer.from_pretrained("lb203/LanguageBind_Image", cache_dir=CACHE_DIR)
    transform = transform_dict["video"](model.modality_config["video"])
    
    # Intention Estimation Module
    intent_pipe = IntentionPipeline() 
    
    # 2. Load Evaluation Dataset
    with open(QUERY_MAP, 'r') as f:
        mapping_data = json.load(f)
    videos = mapping_data.get("videos", {})

    all_results = []

    print(f"\n{'='*85}")
    print("PROPOSED FRAMEWORK EVALUATION: End-to-End Pipeline")
    print(f"{'='*85}")

    for video_file, info in videos.items():
        video_path = os.path.join(VIDEO_DIR, video_file)
        if not os.path.exists(video_path):
            continue

        print(f"\n--- Testing Segment: {video_file} ---")
        
        # Pre-calculate Video Embedding (Back-end)
        vid_tensor = to_device(transform([video_path]), device)
        with torch.no_grad():
            video_emb = F.normalize(model({"video": vid_tensor})["video"], dim=-1)

        for raw_query in info["queries"]:
            try:
                # 1. CHECK THE CACHE FIRST
                # We look for an existing result in all_results to avoid re-calling Gemini
                existing_match = next((r for r in all_results if r['raw_query'] == raw_query), None)

                if existing_match:
                    print(f"  [Cache] Using existing refined prompt: {existing_match['refined_prompt']}")
                    refined_prompt = existing_match['refined_prompt']
                    intent = existing_match['intent']
                else:

                    # ── STEP 1: Intention Estimation Module ──
                    intent_raw = intent_pipe.process(raw_query)
                    intent = json.loads(intent_raw)
                    
                    # ── STEP 2: Prompt Rewriting Module ──
                    # This generates the "Refined Prompt" from your diagram
                    rewriter = PromptRewritingModule(raw_query, intent)
                    refined_prompt = rewriter.genAIsResponse().strip()
                
                # ── STEP 3: LanguageBind & Similarity Matrix ──
                lang_inputs = to_device(tokenizer([refined_prompt], max_length=77, padding="max_length", 
                                                 truncation=True, return_tensors="pt"), device)
                with torch.no_grad():
                    lang_emb = F.normalize(model({"language": lang_inputs})["language"], dim=-1)
                
                # Calculation (Dot Product)
                cos_score = float((video_emb @ lang_emb.T).item())
                
                print(f"  Raw: '{raw_query}'")
                print(f"  Refined: '{refined_prompt}'")
                print(f"  Cosine Score: {cos_score:.4f}")

                all_results.append({
                    "video": video_file,
                    "category": info["category"],
                    "raw_query": raw_query,
                    "refined_prompt": refined_prompt,
                    "intent": intent,
                    "cosine": cos_score
                })

                # API Rate limit safety buffer
                time.sleep(GEMINI_DELAY)

            except Exception as e:
                print(f"  Failed to process query '{raw_query}': {e}")
                continue

    # 3. Final Summary
    with open("eval_robot_proposed_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    mean_cos = statistics.mean([r["cosine"] for r in all_results])
    print(f"\n{'='*85}\nPROPOSED SYSTEM MEAN COSINE: {mean_cos:.4f}\n{'='*85}")

if __name__ == "__main__":
    run_proposed_evaluation()