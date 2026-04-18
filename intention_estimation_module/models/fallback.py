"""
models/fallback.py
Fixes applied:
- All 5 fields now go through fuzzy-match validation in _run_llm_and_parse.
  Previously only task_type and complexity were validated; temporal_context and
  spatial_context skipped get_close_matches entirely, so any imprecise Mistral
  output silently became "none" — causing ~96-99% collapse to "none" for both.
- Removed prompt bias phrases "otherwise simple" and "Use 'none' unless
  explicitly implied", which directly instructed Mistral to default to the
  majority class for complexity and spatial/temporal context.
- Prompt echo stripping: Mistral echoes the full prompt before its answer;
  the JSON extractor now discards everything before the last "Query:" line.
- get_close_matches cutoff lowered to 0.5 for better fuzzy recall.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import re
from difflib import get_close_matches

ALLOWED_OPTIONS = {
    "task_type":        ["temporal localisation", "object detection", "question answering"],
    "output_modality":  ["text", "image", "video"],
    "complexity":       ["temporal", "simple", "causal"],
    "temporal_context": ["none", "during", "before", "after"],
    "spatial_context":  ["none", "inside", "in front", "left of", "right of", "behind", "on top of", "under"],
}


class MistralFallback:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.3"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    # ── Full fallback (classifier not confident) ──────────────────────────────
    def infer_intent(self, text: str) -> dict:
        return self._run_llm_and_parse(self._build_guided_prompt(text), text)

    # ── Partial fallback (task_type already known) ────────────────────────────
    def infer_remaining_intent(self, text: str, fixed_task_type: str) -> dict:
        result = self._run_llm_and_parse(
            self._build_guided_prompt(text, fixed_task_type=fixed_task_type), text
        )
        result["task_type"] = fixed_task_type
        return {
            "output_modality":  result.get("output_modality", "unknown"),
            "complexity":       result.get("complexity", "simple"),
            "temporal_context": result.get("temporal_context", "none"),
            "spatial_context":  result.get("spatial_context", "none"),
        }

    # ── Prompt builder ────────────────────────────────────────────────────────
    def _build_guided_prompt(self, query: str, fixed_task_type: str = None) -> str:
        # FIX: Removed "otherwise simple" and "use none unless explicitly implied".
        # Those two phrases caused Mistral to default to majority classes for
        # complexity (simple) and spatial/temporal context (none) on almost
        # every query, producing ~99% collapse to those values.
        guidelines = """You are an intention extraction model. Output JSON with exactly these keys:
"prompt", "task_type", "output_modality", "complexity", "temporal_context", "spatial_context".

Allowed values only:
- task_type: temporal localisation, object detection, question answering
- output_modality: text, image, video
- complexity: temporal, simple, causal
- temporal_context: none, during, before, after
- spatial_context: none, inside, in front, left of, right of, behind, on top of, under

Rules for task_type:
- "when", "at what point", "find the moment", "locate the segment" → temporal localisation
- "find", "detect", "identify", "locate", "show me where" → object detection
- question asking for reasoning or explanation → question answering

Rules for complexity:
- Involves reasoning about time, sequences, or ordering → temporal
- Involves cause and effect → causal
- Direct lookup with no temporal or causal reasoning → simple

Rules for temporal_context:
- Action happens during a referenced event → during
- Query asks about something before a referenced event → before
- Query asks about something after a referenced event → after
- No reference to another event in time → none

Rules for spatial_context:
- Object is inside a container or region → inside
- Object is in front of something → in front
- Object is to the left → left of
- Object is to the right → right of
- Object is behind something → behind
- Object is on top of something → on top of
- Object is underneath → under
- No spatial relationship referenced → none
"""
        if fixed_task_type:
            guidelines += f'\nThe classifier determined task_type="{fixed_task_type}". Do NOT change it.\n'

        return f"{guidelines}\nQuery: {query}\nRespond with valid JSON only. No text outside the JSON."

    # ── LLM call + parsing ────────────────────────────────────────────────────
    def _run_llm_and_parse(self, prompt: str, text: str) -> dict:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
        )
        raw = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        parsed = {
            "prompt":           text,
            "task_type":        "unknown",
            "output_modality":  "unknown",
            "complexity":       "simple",
            "temporal_context": "none",
            "spatial_context":  "none",
        }

        def _extract_json(s: str) -> str:
            # FIX: Mistral echoes the full prompt before answering.
            # Strip everything up to and including the last "Query:" line
            # before searching for the JSON block.
            if "Query:" in s:
                s = s[s.rfind("Query:"):]
            match = re.search(r"\{.*\}", s, re.DOTALL)
            return match.group() if match else "{}"

        try:
            candidate = json.loads(_extract_json(raw))
            parsed["prompt"] = candidate.get("prompt", text)

            # FIX: Validate ALL fields with fuzzy matching.
            # Previously only task_type and complexity were validated here;
            # temporal_context and spatial_context skipped this block entirely
            # and fell through to a bare .get(..., "none") with no fuzzy match.
            for field in ["task_type", "output_modality", "complexity",
                          "temporal_context", "spatial_context"]:
                val     = str(candidate.get(field, "")).strip().lower()
                allowed = ALLOWED_OPTIONS[field]
                if val in allowed:
                    parsed[field] = val
                else:
                    matches = get_close_matches(val, allowed, n=1, cutoff=0.5)
                    if matches:
                        parsed[field] = matches[0]
                    # else: keep default

        except (json.JSONDecodeError, TypeError):
            pass

        return parsed
