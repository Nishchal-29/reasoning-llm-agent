# Reasoning LLM Agent Framework

A modular, production-ready pipeline for building a Tool-Augmented Reasoning Agent powered by **Qwen/Qwen2.5-3B-Instruct**. This framework optimizes multi-stage training via **Unsloth 4-bit QLoRA** for low-VRAM environments  and serves the final agent through a **Flask REST API** with an interactive web dashboard and Docker containerization.

---

# Architecture

The framework aligns a base instruction-tuned LLM into a robust, multi-tool reasoning agent through a two-phase training curriculum. It is evaluated using a custom ReAct state machine and deployed as an isolated microservice.

```text
Phase A — Supervised Fine-Tuning (Single Combined Curriculum)

┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Reasoning SFT (GSM8K — 7k examples)               │
│  Stage 2: Tool Use SFT (Calc + SymPy + Python — 8k)         │
│  Stage 3: Verification SFT (Multi-tool — 2k)                │
│  Stage 4: Reflection SFT (Error correction — 300)           │
│  ──────────────────────────────────────────────────────     │
│  ↳ Tool Selection (1k) woven throughout                     │
└─────────────────────────────────────────────────────────────┘
                              ↓

Phase B — Group Relative Policy Optimization (GRPO)

┌─────────────────────────────────────────────────────────────┐
│  6 Execution-Aligned Reward Functions                       │
│  (Syntax, Correctness, Execution, Verification,             │
│   Reflection, Format)                                       │
└─────────────────────────────────────────────────────────────┘
                              ↓

Phase C — Evaluation & Agent Loop

┌─────────────────────────────────────────────────────────────┐
│  Live Tool Dispatcher + ReAct State Machine (8 Metrics)     │
└─────────────────────────────────────────────────────────────┘
                              ↓

Phase D — Deployment

┌─────────────────────────────────────────────────────────────┐
│  Dockerized Flask REST API + Gunicorn + Interactive UI      │
└─────────────────────────────────────────────────────────────┘
```

---

# Key Features

## -> Execution-Aligned Training
All tool observations in the training data are computed via live tool execution—never fabricated by the LLM.

## -> Memory-Efficient Optimization
Heavily utilizes Unsloth for 4-bit quantization, allowing complex GRPO and multi-stage SFT to run on a single 16 GB GPU.

## -> Single Combined Curriculum
Avoids sequential adapter merging, preventing quantization noise and catastrophic forgetting under QLoRA.

## -> Safe GRPO Rewards
- `calculator` and `sympy` are executed dynamically during RL.
- `python_repl` uses syntax-only validation (`ast.parse`) to prevent infinite loops during training.

## -> Production Deployment
Includes a Flask REST API (`api.py`) with a built-in interactive chat UI, containerized with Docker and served securely via Gunicorn.

---

# Project Structure

```
reasoning-llm-agent/
│
├── datasets/
│   ├── gsm8k/                        # Raw & processed SFT math datasets
│   └── tool_trajectories/            # Live-execution trajectory data
│       └── generate_trajectories.py   # 7 dedicated generators
│
├── training/
│   ├── sft.py                        # Stage 1: Reasoning SFT (GSM8K)
│   ├── tool_sft.py                   # Stages 2-4: Combined Curriculum SFT
│   └── grpo.py                       # GRPO RL with 6 execution-aligned rewards
│
├── tools/
│   ├── __init__.py                   # Tool registry & dispatch
│   ├── calculator.py                 # Safe AST-based math evaluator
│   ├── python_repl.py               # Sandboxed subprocess code executor
│   ├── sympy_tool.py                # Symbolic algebra solver
│   └── websearch.py                 # DuckDuckGo web search tool
│
├── inference/
│   ├── agent_loop.py                # ReAct state machine (5-loop cap)
|   ├── test_output.py               # Uses agent_loop.py to test each stage of output
│
├── evaluation/
│   └── gsm8k_eval.py               # 8 execution-aligned metrics
│
└── app/
    ├── api.py                       # Flask REST API server
    ├──templates/                    # HTML templates for UI
├── Dockerfile
├── test_api.sh                      # to test Flask REST API
└── requirements.txt
```

---

# Agent & Tool Suite

The core agent operates on a **ReAct (Reason + Act)** loop, generating:

- `<think>` blocks for internal reasoning
- `<tool_call>` blocks for tool interactions
- `<final_answer>` blocks for task completion

The agent is augmented with four deterministic tools:

| Tool | Description |
|--------|-------------|
| Calculator | Safe evaluation for arithmetic |
| SymPy | Symbolic mathematics and algebra |
| Python REPL | Sandboxed Python execution |
| DuckDuckGo Search | Live web retrieval |

---

# Datasets

Training data is dynamically generated.

Instead of extracting trajectories from static datasets, the framework actively executes tools to generate deterministic observation traces.

| Dataset | Count | Stage | Description |
|----------|--------|--------|-------------|
| GSM8K Reasoning | 7,000 | Stage 1 | Foundation reasoning and logic |
| Calculator | 3,000 | Stage 2 | Arithmetic tool usage |
| SymPy | 2,000 | Stage 2 | Algebraic equation solving |
| Python REPL | 2,000 | Stage 2 | Programmatic reasoning |
| Tool Selection | 1,000 | Stage 2 | Context-aware tool selection |
| Verification | 2,000 | Stage 3 | Cross-tool agreement |
| Reflection | 300 | Stage 4 | Error correction patterns |
| **Total** | **~17.3k** | — | Complete curriculum |

---

# Hyperparameters & Unsloth Configuration

## Phase A — SFT Configuration

### LoRA Parameters

```yaml
r: 32
alpha: 64
dropout: 0.05
```

### Target Modules

```python
[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj"
]
```

### Training Settings

```yaml
train_on_responses_only: true
packing: false
```

#### Notes

- Only assistant tokens contribute to loss.
- User and system prompts are masked.
- Packing is disabled to preserve XML/tag structure integrity.

---

## Phase B — GRPO Configuration

Training uses:

```python
NUM_GENERATIONS = 4
```

### Prompt Sampling Distribution

| Dataset Type | Probability |
|-------------|-------------|
| GSM8K | 50% |
| Tool Usage | 20% |
| Verification | 15% |
| Reflection | 15% |

---

## Reward Functions

| Reward | Score |
|----------|--------|
| Format Check | +0.25 |
| Tool Syntax | +0.25 |
| Tool Execution | +1.00 |
| Verification Match | +1.00 |
| Reflection | +1.00 |
| Correctness | +1.00 |

### Format Check (+0.25)

Validates proper ordering:

```xml
<think>
<tool_call>
<final_answer>
```

### Tool Syntax (+0.25)

Validates parseable JSON within tool calls.

### Tool Execution (+1.00)

Executes the requested tool and compares the observed output against the model's prediction.

### Verification Match (+1.00)

Rewards reaching the same conclusion using multiple independent tools.

### Reflection (+1.00)

Rewards successful error recovery:

```text
Error
  ↓
Correction
  ↓
Success
```

### Correctness (+1.00)

Checks final answer equivalence against ground truth.

---

# Training Pipeline

## 1. Generate Tool Trajectories

Generate synthetic data through live tool execution.

```bash
python -m datasets.tool_trajectories.generate_trajectories
```

---

## 2. Reasoning SFT (Stage 1)

Train the foundational reasoning format.

```bash
python training/sft.py
```

---

## 3. Combined Curriculum SFT (Stages 2–4)

Merge the reasoning adapter into base weights and continue training on the full interactive curriculum.

```bash
python training/tool_sft.py 
```

---

## 4. GRPO Training (Phase B)

Optimize tool usage, verification, and reflection.

```bash
python training/grpo.py 
```

---

## 5. Evaluation

Evaluate on GSM8K.

```bash
python evaluation/gsm8k_eval.py
```

---

# Deployment & API

The trained agent is served through a Flask REST application (`api.py`).

Responsibilities include:

- Dynamic VRAM allocation
- ReAct loop orchestration
- Tool execution
- Front-end rendering

---

# Running Locally (Development)

## Gunicorn (Recommended)

```bash
gunicorn \
  --workers=1 \
  --threads=1 \
  --timeout=300 \
  --bind 0.0.0.0:5000 \
  wsgi:app
```

## Standard Python

```bash
python wsgi.py
```

Open the web dashboard:

```text
http://localhost:5000
```

---

# Docker Containerization (Production)

## 1. Build the Docker Image

Ensure `.dockerignore` excludes:

- Raw datasets
- Cache directories
- Training artifacts

```bash
docker build -t reasoning-agent:v1 .
```

---

## 2. Run the Container

```bash
docker run -d \
  --name qwen-agent \
  --gpus all \
  -p 5000:5000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  reasoning-agent:v1
```

---

# API Endpoints

| Method | Endpoint | Description |
|----------|----------|-------------|
| GET | `/` | Interactive web UI |
| GET | `/api` | API documentation |
| POST | `/api/query` | Execute full ReAct loop |
| POST | `/api/tool` | Execute a single tool |
| GET | `/api/tools` | Available tool schemas |
| GET | `/api/health` | Health check + tool pipeline validation |
| GET | `/api/model/info` | Runtime model configuration and VRAM stats |

---

# Example API Request

```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{
        "query": "Calculate the volume of a sphere with radius 4.5 using python."
      }'
```

Or can use test_api.sh to test all the Flask REST APIs

---
