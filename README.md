# reasoning-llm-agent

This project presents **reasoning-llm-agent**, an Agentic AI system created for the Saptang Labs Machine Learning Challenge to address the common failure of Large Language Models in multi-step reasoning tasks.  
The system works by decomposing complex logic questions into a sequence of simpler sub-tasks.  
It then autonomously selects and executes the appropriate tool, such as a calculator or code interpreter, for each step.  
This modular approach ensures accuracy and produces a transparent, human-readable reasoning trace for each final answer.

---

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
  - [Preflight Checks](#preflight-checks)
  - [Recommended Installation (uv)](#recommended-installation-uv)
  - [Fallback: uv add](#fallback-uv-add)
  - [Conda Alternative](#conda-alternative)
  - [CPU-only Users](#cpu-only-users)
- [Verification](#verification)
- [Running Scripts](#running-scripts)
- [Requirements.txt](#requirementstxt)

---

## Requirements

- Python 3.10+
- GPU with CUDA 12+ recommended for full performance
- NVIDIA drivers installed (`nvidia-smi` to check)
- `uv` package for environment management

---

## Installation

### Preflight Checks

```bash
which python
python -V
nvidia-smi || true  # checks for GPU and drivers
Recommended Installation (uv)
bash
Copy code
# install uv if needed
python -m pip install --upgrade pip
python -m pip install uv

# create virtualenv and install dependencies from pyproject.toml
UV_HTTP_TIMEOUT=300 uv run
Run scripts:

bash
Copy code
# activate venv
source .venv/bin/activate
python agent/check.py      # or python main.py
# or without activating
./.venv/bin/python agent/check.py
Notes:

UV_HTTP_TIMEOUT=300 is recommended for slow networks.

uv run installs torch and all other dependencies automatically.

Fallback: uv add
If uv run fails to install torch:

bash
Copy code
# ensure uv is installed
python -m pip install --upgrade pip
python -m pip install uv

# pre-fetch heavy packages
UV_HTTP_TIMEOUT=600 uv add "torch==2.8.0"

# then run normally
UV_HTTP_TIMEOUT=300 uv run

# run your script
./.venv/bin/python agent/check.py
Conda Alternative (robust for GPU)
bash
Copy code
conda create -n reasoning-gpu python=3.10 -y
conda activate reasoning-gpu

# install CUDA + cuDNN runtime libraries
conda install -c nvidia cudatoolkit=12.5 cudnn=9.10 -y

# install project
python -m pip install --upgrade pip
python -m pip install -e .

# run
python agent/check.py
CPU-only Users
bash
Copy code
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.8.0"

# create environment with uv
uv run

# run
./.venv/bin/python agent/check.py
Verification
bash
Copy code
python - <<'PY'
import importlib, sys
print("python:", sys.executable)
print("torch installed:", bool(importlib.util.find_spec("torch")))
try:
    import torch
    print("torch:", torch.__version__)
    print("torch.version.cuda (build):", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    print("cudnn:", torch.backends.cudnn.version())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch import failed:", e)
PY
Running Scripts
bash
Copy code
# inside venv
python agent/check.py
python main.py
Or without activating venv:

bash
Copy code
./.venv/bin/python agent/check.py
Requirements.txt
A requirements.txt is included for users who prefer pip:

bash
Copy code
# install dependencies using pip
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
Generated from the .venv created by uv; includes torch==2.8.0+cu128 and all other dependencies.

Notes
Recommend using a GPU for full functionality.

uv run is the primary workflow; fallback with uv add if needed.

Set UV_HTTP_TIMEOUT high (300–600) for slow networks.

Optionally, maintainers can use a bootstrap.sh to pre-resolve heavy packages:

bash
Copy code
#!/usr/bin/env bash
set -e
UV_HTTP_TIMEOUT=600 uv add "torch==2.8.0" torchvision torchaudio
UV_HTTP_TIMEOUT=300 uv run
yaml
Copy code

---

If you want, I can **also generate a ready-to-commit `requirements.txt`** for this repo with all your `.venv` packages pinned, including `torch==2.8.0+cu128`, so users can just `pip install -r requirements.txt` without running `uv`.  

Do you want me to do that next?