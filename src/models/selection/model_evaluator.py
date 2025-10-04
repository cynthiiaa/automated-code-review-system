from typing import Dict, List
import torch
from transformers import AutoModelForCausalLM
import time

class ModelEvaluator:
    """Compare different models"""

    MODELS_TO_TEST = [
        "codellama/CodeLlama-7b-hf",
        "microsoft/codebert-base",
        "Salesforce/codet5p-770m",
        "bigcode/starcode2-3b"
    ]

    def benchmark_models(self, test_samples: List[Dict]) -> Dict:
        results = {}

        for model_name in self.MODELS_TO_TEST:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )

            metrics = self._evaluate_model(model, test_samples)
            results[model_name] = metrics

        return results
    
    def _evaluate_model(self, model, samples) -> Dict:
        metrics = {
            "latency_p50": [],
            "latency_p95": [],
            "tokens_per_second": [],
            "memory_usage": [],
            "quality_score": []
        }

        for sample in samples:
            start = time.time()
            output = model.generate(sample["input_ids"], max_new_tokens=256)
            latency = time.time() - start

            metrics["latency_p50"].append(latency)
        
        return metrics