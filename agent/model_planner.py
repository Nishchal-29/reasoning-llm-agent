from __future__ import annotations
import json
import re
from typing import List, Dict, Any, Optional, Tuple
import os
from transformers import T5ForConditionalGeneration, T5TokenizerFast
import torch

def create_planner_prompt(question: str, prompt_template_path: str) -> str:
    with open(prompt_template_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    final_prompt = prompt_template.replace("{question}", question)

    return final_prompt

# Lightweight LLM planner wrapper (imports transformers lazily so module import is cheap)
class PlannerLLM:
    def __init__(self, model_name: str = 'google/flan-t5-small', device: Optional[str] = None, cache_dir: Optional[str] = None):
        self.model_name = model_name
        # detect device simply — allow user override
        self.device = device or ('cuda' if (os.environ.get('CUDA_VISIBLE_DEVICES') or False) else 'cpu')
        self.cache_dir = cache_dir
        self.model = None
        self.tokenizer = None

    def load_model(self):
        self.tokenizer = T5TokenizerFast.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        self.model.to(self.device)

    def generate(self, prompt: str,
        max_new_tokens: int = 256,
        num_return_sequences: int = 3,
        temperature: float = 0.0,
        top_p: float = 1.0,
        num_beams: int = 4,
        repetition_penalty: float = 1.2,
        early_stopping: bool = True) -> List[Dict[str, Any]]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError('Model not loaded. Call load_model() first.')

        # Use tokenizer to prepare inputs
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024).to(self.device)

        # If using beam search, num_return_sequences must be <= num_beams
        num_return_sequences = min(num_return_sequences, num_beams)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            do_sample=False,              # greedy / beam search, not sampling
            repetition_penalty=repetition_penalty,
            early_stopping=early_stopping,
            return_dict_in_generate=True,
            output_scores=False
        )

        outputs = self.model.generate(**inputs, **gen_kwargs)

        results = []
        for i in range(len(outputs.sequences)):
            text = self.tokenizer.decode(outputs.sequences[i], skip_special_tokens=True)
            # Post-process: try to extract the first valid JSON array/object
            parsed = self._extract_json_from_text(text)
            # If parsed is None, attempt repair heuristics
            if parsed is None:
                repaired = self._repair_and_extract(text)
                parsed = repaired
            results.append({'raw': text, 'parsed': parsed})
        return results

    def _extract_json_from_text(self, s: str) -> Optional[Any]:
        s = s.strip()
        first = None
        for i, ch in enumerate(s):
            if ch in '{[':
                first = i
                break
        if first is None:
            return None
        stack = []
        pairs = {'{':'}', '[':']'}
        for j in range(first, len(s)):
            ch = s[j]
            if ch in pairs:
                stack.append(pairs[ch])
            elif stack and ch == stack[-1]:
                stack.pop()
                if not stack:
                    candidate = s[first:j+1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        try:
                            cand2 = candidate.replace("'", '"')
                            return json.loads(cand2)
                        except Exception:
                            return None
        return None
    
    def _repair_and_extract(self, s: str) -> Optional[Any]:
        s = s.strip()
        # find first '['
        idx = s.find('[')
        if idx == -1:
            # maybe model started with '{' (single object). try to extract {...}
            idx_obj = s.find('{')
            if idx_obj != -1:
                # attempt to extract balanced {...}
                parsed = self._try_extract_braced(s, '{', '}')
                if parsed is not None:
                    return parsed
                # otherwise wrap the content in [ ... ] and try parse
                candidate = '[' + s + ']'
                try:
                    return json.loads(candidate)
                except Exception:
                    return None
            return None

        # find matching bracket by counting
        stack = 0
        end = None
        for i in range(idx, len(s)):
            if s[i] == '[':
                stack += 1
            elif s[i] == ']':
                stack -= 1
                if stack == 0:
                    end = i
                    break
        if end is None:
            # append ']' and try
            candidate = s[idx:] + ']'
        else:
            candidate = s[idx:end+1]
        # sanitize common single-quote usage
        cand2 = candidate
        try:
            return json.loads(cand2)
        except Exception:
            try:
                cand3 = cand2.replace("'", '"')
                return json.loads(cand3)
            except Exception:
                return None

    def _try_extract_braced(self, s: str, open_ch: str, close_ch: str) -> Optional[Any]:
        """Find a balanced {...} substring and try json.loads with repairs."""
        start = s.find(open_ch)
        if start == -1:
            return None
        stack = 0
        end = None
        for i in range(start, len(s)):
            if s[i] == open_ch:
                stack += 1
            elif s[i] == close_ch:
                stack -= 1
                if stack == 0:
                    end = i
                    break
        if end is None:
            return None
        candidate = s[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            try:
                return json.loads(candidate.replace("'", '"'))
            except Exception:
                return None


    def close(self):
        if self.model is not None:
            try:
                del self.model
            except Exception:
                pass
        if self.tokenizer is not None:
            try:
                del self.tokenizer
            except Exception:
                pass
