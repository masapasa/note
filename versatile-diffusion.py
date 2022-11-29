from diffusers import VersatileDiffusionPipeline
import torch
import requests
from io import BytesIO
from PIL import Image

pipe = VersatileDiffusionPipeline.from_pretrained("shi-labs/versatile-diffusion", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# initial image
url = "https://huggingface.co/datasets/diffusers/images/resolve/main/benz.jpg"
response = requests.get(url)
image = Image.open(BytesIO(response.content)).convert("RGB")

# prompt
prompt = "a red cat on the green car"

# text to image
image = pipe.text_to_image(prompt).images[0]

# image variation
image = pipe.image_variation(image).images[0]

# image variation
image = pipe.dual_guided(prompt, image).images[0]
image.save("car-cat.png")