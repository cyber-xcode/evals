from typing import Any

import evals
import evals.metrics
from evals.api import CompletionFn
from evals.elsuite import utils


class BatchIncludes(evals.Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        samples_jsonl: str,
        ignore_case: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) == 1, "Includes only supports one completion fn"
        self.samples_jsonl = samples_jsonl
        self.ignore_case = ignore_case
        
        self.temperature = kwargs.get("temperature",0.0)
        self.top_p = kwargs.get("top_p",1.0)
        self.sample_n = kwargs.get("n",1)
        # self.top_k = kwargs.get("top_k",-1)
        

    def eval_sample(self, sample: Any, *_):
        assert isinstance(sample, dict), "sample must be a dict"
        assert "input" in sample, "sample must have an 'input' key"
        assert "ideal" in sample, "sample must have an 'ideal' key"

        prompt = sample["input"]
        try :
            result = self.completion_fn(
                prompt=prompt,
                
                # top_k=self.top_k,
                top_p=self.top_p,
                temperature=self.temperature,
                n=self.sample_n,
                max_tokens=2048,
            )
        except :
            return None 
        all_samples = result.get_completions() 
        
        sampled = all_samples[0]
        
        if 1==1:
    
            ideal = sample["ideal"]
            if not isinstance(ideal, list):
                ideal = [ideal]
    
            assert isinstance(ideal, list) and all(
                isinstance(i, str) for i in ideal
            ), "ideal must be a list of strings"
    
            includes_answer = any(
                [utils.get_answer(sampled, ref, self.ignore_case) is not None for ref in ideal]
            )
            evals.record.record_match(
                includes_answer, expected=sample["ideal"], picked=all_samples, sampled=sampled
            )
        return includes_answer

    def run(self, recorder):
        samples = self.get_samples()
        self.eval_all_samples(recorder, samples)
        events = recorder.get_events("match")
        return {
            "accuracy": evals.metrics.get_accuracy(events),
            "boostrap_std": evals.metrics.get_bootstrap_accuracy_std(events),
        }
