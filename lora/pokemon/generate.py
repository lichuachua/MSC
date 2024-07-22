from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("./fine-tune-lora", torch_dtype=torch.float16, use_safetensors=True).to("cuda")
image = pipeline("A pokemon with blue eyes", num_inference_steps=50, guidance_scale=7.5).images[0]
image.save("pokemon.png")

