"""
Patched generate_teacher_plans.py
- Uses batch generation
- Proper BitsAndBytesConfig (new HF API) for 8-bit quant
- fp16 fallback
- use_fast tokenizer
- tqdm progress bar and per-batch timing debug prints
- option to skip execution (debugging) via --skip_exec
- writes same jsonl output schema
"""
from string import Template
import argparse, json, time, os
from pathlib import Path
from typing import Optional, List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from tqdm import tqdm

try:
    from agent.executor import execute_plan
except Exception:
    execute_plan = None


def load_teacher_model(model_name, local_dir=None, token: Optional[str]=None, dtype=torch.float16,
                       use_8bit=True, offload_folder: Optional[str]=None):
    if local_dir:
        model_name = local_dir

    # faster tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    load_kwargs = dict(trust_remote_code=True, low_cpu_mem_usage=True)
    if token:
        load_kwargs["use_auth_token"] = token

    if use_8bit:
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                quantization_config=bnb_config,
                offload_folder=offload_folder,
                **load_kwargs
            )
            model.eval()
            print("Loaded model in 8-bit with BitsAndBytesConfig.")
            return tokenizer, model
        except Exception as e:
            print("8-bit + BitsAndBytesConfig load failed, falling back to fp16. Error:", e)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=dtype,
            offload_folder=offload_folder,
            **load_kwargs
        )
        model.eval()
        print("Loaded model in fp16.")
        return tokenizer, model
    except Exception as e:
        print("FP16 loading failed; trying CPU fp32 as last resort. Error:", e)

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    print("Loaded model on CPU (fp32).")
    return tokenizer, model


def extract_json_from_text(s: str):
    """Extract first top-level JSON object/array from text, return parsed object or None."""
    if s is None:
        return None
    s = s.strip()
    first = None
    for i, ch in enumerate(s):
        if ch in '[{':
            first = i
            break
    if first is None:
        return None
    stack = []
    pairs = {'{': '}', '[': ']'}
    for j in range(first, len(s)):
        ch = s[j]
        if ch in pairs:
            stack.append(pairs[ch])
        elif stack and ch == stack[-1]:
            stack.pop()
            if not stack:
                cand = s[first:j + 1]
                try:
                    return json.loads(cand)
                except Exception:
                    try:
                        cand2 = cand.replace("'", '"')
                        return json.loads(cand2)
                    except Exception:
                        return None
    return None


def contains_answer_step(parsed_plan) -> bool:
    """Heuristic: check if any step's input contains a return with 'answer' or pattern returning answer dict."""
    if not isinstance(parsed_plan, (list, tuple)):
        return False
    for step in parsed_plan:
        if not isinstance(step, dict):
            continue
        inp = (step.get("input") or "").lower()
        if "return" in inp and "answer" in inp:
            return True
        if step.get("tool") == "python" and "return" in inp:
            return True
    return False


def build_prompt(tmpl: Template, problem: str, problem_type: str) -> str:
    subs = {"problem_type": problem_type, "question": problem}
    return tmpl.substitute(subs)


def batch_generate(prompts: List[str], tokenizer, model, gen_args: dict, max_input_length: int) -> List[str]:
    inputs = tokenizer(prompts, return_tensors="pt", truncation=True,
                       max_length=max_input_length, padding=True)
=    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(**inputs, **gen_args)
    texts = [tokenizer.decode(out, skip_special_tokens=False) for out in outputs]
    return texts


def safe_exec_plan(parsed):
    if execute_plan is None:
        return {"ok": False, "error": "execute_plan not available (module import failed)"}
    try:
        res = execute_plan(parsed)
        return res
    except Exception as e:
        return {"ok": False, "error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--prompt_template", default="prompts/planner_prompt.txt")
    ap.add_argument("--teacher_model", default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--local_model_dir", default=None, help="Optional local path to model weights")
    ap.add_argument("--out", default="data/teacher_plans.jsonl")
    ap.add_argument("--token", default=None, help="HuggingFace token")
    ap.add_argument("--use_auth_token", action="store_true", help="Try HF_TOKEN env var if --token not set")
    ap.add_argument("--num_return_sequences", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sleep_secs", type=float, default=0.05)
    ap.add_argument("--batch_size", type=int, default=8, help="Batch size for generation (adjust for memory)")
    ap.add_argument("--use_8bit", action="store_true", help="Attempt load_in_8bit via bitsandbytes")
    ap.add_argument("--offload_folder", default=None, help="Folder to offload weights if using device_map")
    ap.add_argument("--skip_exec", action="store_true", help="Skip execute_plan (useful for debugging slow execution)")
    args = ap.parse_args()

    token = args.token
    if token is None and args.use_auth_token:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    prompt_template_text = open(args.prompt_template, "r", encoding="utf-8").read()
    tmpl = Template(prompt_template_text)
    print("Loading teacher model/tokenizer (this may take a while)...")
    tokenizer, model = load_teacher_model(args.teacher_model, local_dir=args.local_model_dir,
                                          token=token, dtype=torch.float16,
                                          use_8bit=args.use_8bit, offload_folder=args.offload_folder)
    print("Loaded model:", args.teacher_model)

    do_sample = float(args.temperature) > 0.0
    gen_args = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=do_sample,
        temperature=float(args.temperature) if do_sample else 1.0,
        top_p=float(args.top_p) if do_sample else 1.0,
        num_return_sequences=max(1, args.num_return_sequences),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    if hasattr(model.config, "max_position_embeddings") and model.config.max_position_embeddings:
        max_input_length_default = max(256, model.config.max_position_embeddings - gen_args.get("max_new_tokens", 128))
    else:
        max_input_length_default = 1024

    outf = open(args.out, "w", encoding="utf-8")
    total_lines = sum(1 for _ in open(args.input_jsonl, 'r', encoding='utf-8'))
    pbar = tqdm(total=total_lines, desc="Problems", unit="p")

    with open(args.input_jsonl, "r", encoding="utf-8") as fin:
        batch_prompts = []
        batch_meta: List[Tuple[str, str, str]] = []  
        total = 0
        for line_idx, line in enumerate(fin):
            rec = json.loads(line)
            problem = rec.get("problem", "")
            ptype = rec.get("problem_type", "general")
            prompt = build_prompt(tmpl, problem, ptype)
            batch_prompts.append(prompt)
            batch_meta.append((problem, ptype, prompt))

            if len(batch_prompts) >= args.batch_size:
                t0 = time.time()
                try:
                    texts = batch_generate(batch_prompts, tokenizer, model, gen_args, max_input_length_default)
                except Exception as e:
                    print(f"Generation failed on batch ending at line {line_idx}: {e}")
                    texts = [""] * len(batch_prompts)
                t_batch = time.time() - t0
                print(f"[DEBUG] batch of {len(batch_prompts)} gen time: {t_batch:.3f}s")

                stride = args.num_return_sequences
                if stride == 1:
                    chosen_texts = texts
                else:
                    chosen_texts = []
                    for i in range(0, len(texts), stride):
                        chosen_texts.append(texts[i])

                for (problem, ptype, _prompt), raw in zip(batch_meta, chosen_texts):
                    parsed = extract_json_from_text(raw)
                    verified = False
                    exec_result = None
                    if parsed is not None and isinstance(parsed, list) and contains_answer_step(parsed):
                        if args.skip_exec:
                            exec_result = {"ok": False, "error": "skipped execution (debug)"}
                        else:
                            exec_result = safe_exec_plan(parsed)
                            verified = bool(exec_result.get("ok", False))
                    else:
                        exec_result = {"ok": False, "error": "no valid parsed plan or no answer-returning step found"}

                    outrec = {
                        "problem": problem,
                        "problem_type": ptype,
                        "teacher_raw": raw,
                        "parsed_plan": parsed,
                        "verified": verified,
                        "exec_result": exec_result
                    }
                    outf.write(json.dumps(outrec, ensure_ascii=False) + "\n")
                    outf.flush()
                    total += 1
                    pbar.update(1)

                batch_prompts = []
                batch_meta = []
                time.sleep(args.sleep_secs)

        if batch_prompts:
            t0 = time.time()
            try:
                texts = batch_generate(batch_prompts, tokenizer, model, gen_args, max_input_length_default)
            except Exception as e:
                print("Final batch generation failed:", e)
                texts = [""] * len(batch_prompts)
            t_batch = time.time() - t0
            print(f"[DEBUG] final batch of {len(batch_prompts)} gen time: {t_batch:.3f}s")

            stride = args.num_return_sequences
            if stride == 1:
                chosen_texts = texts
            else:
                chosen_texts = []
                for i in range(0, len(texts), stride):
                    chosen_texts.append(texts[i])

            for (problem, ptype, _prompt), raw in zip(batch_meta, chosen_texts):
                parsed = extract_json_from_text(raw)
                verified = False
                exec_result = None
                if parsed is not None and isinstance(parsed, list) and contains_answer_step(parsed):
                    if args.skip_exec:
                        exec_result = {"ok": False, "error": "skipped execution (debug)"}
                    else:
                        exec_result = safe_exec_plan(parsed)
                        verified = bool(exec_result.get("ok", False))
                else:
                    exec_result = {"ok": False, "error": "no valid parsed plan or no answer-returning step found"}

                outrec = {
                    "problem": problem,
                    "problem_type": ptype,
                    "teacher_raw": raw,
                    "parsed_plan": parsed,
                    "verified": verified,
                    "exec_result": exec_result
                }
                outf.write(json.dumps(outrec, ensure_ascii=False) + "\n")
                outf.flush()
                total += 1
                pbar.update(1)

    outf.close()
    pbar.close()
    print(f"Done. Saved {total} records to {args.out}")

if __name__ == "__main__":
    main()