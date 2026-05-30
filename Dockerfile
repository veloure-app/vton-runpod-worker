FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Clonează IDM-VTON
RUN git clone https://github.com/yisol/IDM-VTON /workspace/IDM-VTON

WORKDIR /workspace/IDM-VTON

RUN pip install -r requirements.txt
RUN pip install runpod

# Descarcă modelul la build time
RUN pip install huggingface_hub
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('yisol/IDM-VTON', local_dir='/workspace/models/IDM-VTON')"

COPY handler.py /workspace/handler.py

WORKDIR /workspace

CMD ["python", "handler.py"]
