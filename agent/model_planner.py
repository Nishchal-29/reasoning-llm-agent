from __future__ import annotations
import json
import re
from typing import List, Dict, Any, Optional, Tuple
import os

def format_few_shot_examples(examples: List[Dict[str, Any]]) -> str:
    """
    Formats the few-shot examples into a string for the prompt.
    Defensive: supports examples that use keys 'question' or 'problem' and
    gracefully falls back when 'plan' is missing.
    """
    formatted_str = ""
    for example in examples:
        q = example.get('question') or example.get('problem') or example.get('prompt') \
            or example.get('problem_statement') or '<MISSING_QUESTION>'
        formatted_str += f"Problem: {q}\n"

        plan = example.get('plan')
        if plan is None:
            # Fall back: if only an answer is provided, fake a one-step plan
            ans = example.get('answer')
            if ans is not None:
                # jsonify the answer properly
                try:
                    ans_json = json.dumps(ans, ensure_ascii=False)
                except Exception:
                    ans_json = str(ans)
                plan = [f"Return {ans_json}"]
            else:
                plan = ["<no plan available>"]

        # Normalize to list
        if not isinstance(plan, list):
            plan = [plan]

        # If plan items are dicts -> pretty JSON, else join strings
        if all(isinstance(p, dict) for p in plan):
            plan_str = json.dumps(plan, ensure_ascii=False, indent=2)
            plan_str = '\n'.join('  ' + line for line in plan_str.splitlines())
            formatted_str += "Plan:\n" + plan_str + "\n---\n"
        else:
            # join plan items (string steps)
            plan_str = "\n".join(str(p) for p in plan)
            formatted_str += "Plan:\n" + plan_str + "\n---\n"

    return formatted_str


def create_planner_prompt(question: str, prompt_template_path: str, few_shot_path: str, max_examples: Optional[int] = None) -> str:
    """
    Create the final prompt in a safe way (avoid str.format with unescaped braces).
    The prompt template should include the literal substrings:
      {few_shot_examples}
      {question}
    which will be replaced via str.replace (safe for JSON content).
    """
    # Load template text
    with open(prompt_template_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    # Load few-shot examples (must be a JSON list)
    with open(few_shot_path, 'r', encoding='utf-8') as f:
        few_shot_examples = json.load(f)

    if not isinstance(few_shot_examples, list):
        raise ValueError('Few-shot file must contain a JSON list of examples.')

    if max_examples is not None:
        few_shot_examples = few_shot_examples[:max_examples]

    formatted_examples = format_few_shot_examples(few_shot_examples)

    # IMPORTANT: use replace instead of format to avoid KeyError from braces in JSON
    final_prompt = prompt_template.replace("{few_shot_examples}", formatted_examples)
    final_prompt = final_prompt.replace("{question}", question)

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
        """Lazily import and load the tokenizer & model. Call this when ready to use GPU/CPU."""
        from transformers import T5ForConditionalGeneration, T5TokenizerFast
        import torch

        self.tokenizer = T5TokenizerFast.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        self.model.to(self.device)

    def generate(self, prompt: str, max_length: int = 512, num_return_sequences: int = 5, temperature: float = 0.7, top_p: float = 0.9) -> List[Dict[str, Any]]:
        """Generate candidate outputs for the provided prompt."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError('Model not loaded. Call load_model() first.')

        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024).to(self.device)
        gen_kwargs = dict(
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            max_length=max_length,
        )
        outputs = self.model.generate(**inputs, **gen_kwargs)
        results = []
        for out in outputs:
            text = self.tokenizer.decode(out, skip_special_tokens=True)
            parsed = self._extract_json_from_text(text)
            results.append({'raw': text, 'parsed': parsed})
        return results

    def _extract_json_from_text(self, s: str) -> Optional[Any]:
        """Try extracting a JSON object or array from model output."""
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
