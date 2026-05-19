# """
# models/pipeline.py
# """

# import json
# import os
# from models.classifier import IntentionClassifier
# from models.fallback import MistralFallback

# MODEL_NAME = os.path.join(
#     os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
#     "deberta-v3-large"
# )

# class IntentionPipeline:
#     def __init__(self):
#         self.classifier = IntentionClassifier(
#             model_name=MODEL_NAME,                          # FIX: local path, not HuggingFace
#             training_data_path="data/ground_truth.jsonl",
#             peft_path="models/fine_tuned_classifier",
#             confidence_threshold=0.4,
#         )
#         self.fallback = MistralFallback()

#     process(self, query: str) -> str:
#         task_pred            = self.classifier.predict_task_type(query)
#         confidence           = task_pred["confidence"]
#         task_type            = task_pred["task_type"]
#         secondary_task       = task_pred.get("secondary_task_type", None)
#         secondary_confidence = task_pred.get("secondary_confidence", 0.0)

#         use_fallback     = self.classifier.should_use_fallback(confidence, task_type=task_type)
#         task_type_spaced = task_type.replace("_", " ").lower()

#         print(f"[Classifier] task_type='{task_type}' confidence={confidence:.3f} "
#               f"fallback={'yes' if use_fallback else 'no'}")

#         if use_fallback:
#             print("⚠️  Low confidence — using Mistral for full classification.")
#             mistral_result = self.fallback.infer_intent(query)
#         else:
#             print("✅ Confident — using Mistral for remaining fields only.")
#             # Let Mistral handle the complex rewrite while retaining DeBERTa's fine-tuned configurations
#             mistral_result = self.fallback.infer_remaining_intent(
#                 query, fixed_task_type=task_type_spaced
#             )

#         has_secondary = secondary_task and secondary_task != task_type

#         final_result = {
#             "prompt":               query,
#             "task_type":            mistral_result.get("task_type", task_type_spaced) if use_fallback else task_type_spaced,
#             "confidence":           round(confidence, 4),
#             "secondary_task_type":  secondary_task if has_secondary else None,
#             "secondary_confidence": round(secondary_confidence, 4) if has_secondary else None,
#             "overall_confidence":   round((confidence + secondary_confidence) / 2 if has_secondary else confidence, 4),
#             # FIXED: Dynamically merge your fine-tuned DeBERTa values into your outputs!
#             "complexity":           task_pred["complexity"] if not use_fallback else mistral_result.get("complexity", "simple"),
#             "temporal_context":     task_pred["temporal_context"] if not use_fallback else mistral_result.get("temporal_context", "none"),
#             "spatial_context":      task_pred["spatial_context"] if not use_fallback else mistral_result.get("spatial_context", "none"),
#             **{k: v for k, v in mistral_result.items() if k not in ["task_type", "confidence", "complexity", "temporal_context", "spatial_context"]}
#         }

#         return json.dumps(final_result, indent=2)

"""
models/pipeline.py
"""

import json
import os
from models.classifier import IntentionClassifier
from models.fallback import MistralFallback

MODEL_NAME = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "deberta-v3-large"
)

class IntentionPipeline:
    def __init__(self):
        self.classifier = IntentionClassifier(
            model_name=MODEL_NAME,                          # FIX: local path, not HuggingFace
            training_data_path="data/ground_truth.jsonl",
            peft_path="models/fine_tuned_classifier",
            confidence_threshold=0.4,
        )
        self.fallback = MistralFallback()

    def process(self, query: str) -> str:
        task_pred            = self.classifier.predict_task_type(query)
        confidence           = task_pred["confidence"]
        task_type            = task_pred["task_type"]
        secondary_task       = task_pred.get("secondary_task_type", None)
        secondary_confidence = task_pred.get("secondary_confidence", 0.0)

        # FIX: store result once — avoids double-call on borderline confidence
        # FIX: pass task_type so per-class thresholds are actually used
        use_fallback     = self.classifier.should_use_fallback(confidence, task_type=task_type)
        task_type_spaced = task_type.replace("_", " ").lower()

        print(f"[Classifier] task_type='{task_type}' confidence={confidence:.3f} "
              f"fallback={'yes' if use_fallback else 'no'}")

        if use_fallback:
            print("⚠️  Low confidence — using Mistral for full classification.")
            mistral_result = self.fallback.infer_intent(query)
        else:
            print("✅ Confident — using Mistral for remaining fields only.")
            mistral_result = self.fallback.infer_remaining_intent(
                query, fixed_task_type=task_type_spaced
            )

        has_secondary = secondary_task and secondary_task != task_type

        final_result = {
            "prompt":               query,
            "task_type":            mistral_result.get("task_type", task_type_spaced) if use_fallback else task_type_spaced,
            "confidence":           round(confidence, 4),
            "secondary_task_type":  secondary_task if has_secondary else None,
            "secondary_confidence": round(secondary_confidence, 4) if has_secondary else None,
            "overall_confidence":   round((confidence + secondary_confidence) / 2 if has_secondary else confidence, 4),
            **{k: v for k, v in mistral_result.items() if k not in ["task_type", "confidence"]}
        }

        return json.dumps(final_result, indent=2)