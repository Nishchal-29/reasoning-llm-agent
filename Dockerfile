FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install "unsloth[cu121-torch220] @ git+https://github.com/unslothai/unsloth.git"
RUN pip install --no-deps trl peft accelerate bitsandbytes xformers

COPY . .
EXPOSE 5000
CMD ["gunicorn", "--workers=1", "--threads=1", "--timeout=300", "--bind", "0.0.0.0:5000", "wsgi:app"]