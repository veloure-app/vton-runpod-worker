FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y git git-lfs && rm -rf /var/lib/apt/lists/*

# Clonează IDM-VTON
RUN git clone https://github.com/yisol/IDM-VTON /workspace/IDM-VTON

# Instalează dependențele manual (fără requirements.txt)
RUN pip install --no-cache-dir \
    accelerate \
    diffusers \
    transformers \
    torchvision \
    einops \
    Pillow \
    opencv-python-headless \
    huggingface_hub \
    runpod \
    requests

# Descarcă modelul
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('yisol/IDM-VTON', local_dir='/workspace/models/IDM-VTON')"

COPY handler.py /workspace/handler.py

WORKDIR /workspace

CMD ["python", "handler.py"]
