import runpod
import requests
import torch
import base64
from PIL import Image
from io import BytesIO
import sys

sys.path.append("/workspace/IDM-VTON")

pipe = None

def load_model():
    global pipe
    if pipe is not None:
        return
    from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
    pipe = TryonPipeline.from_pretrained(
        "/workspace/models/IDM-VTON",
        torch_dtype=torch.float16
    ).to("cuda")

def download_image(url):
    response = requests.get(url, timeout=30)
    return Image.open(BytesIO(response.content)).convert("RGB")

def handler(job):
    try:
        load_model()
        input_data = job["input"]
        human_url = input_data["human_image_url"]
        garment_url = input_data["garment_image_url"]
        category = input_data.get("category", "upper_body")

        human_img = download_image(human_url)
        garment_img = download_image(garment_url)

        result = pipe(
            human_image=human_img,
            garment_image=garment_img,
            category=category,
            num_inference_steps=30,
        ).images[0]

        buffer = BytesIO()
        result.save(buffer, format="JPEG", quality=90)
        img_b64 = base64.b64encode(buffer.getvalue()).decode()

        return { "image_base64": img_b64 }

    except Exception as e:
        return { "error": str(e) }

runpod.serverless.start({ "handler": handler })
