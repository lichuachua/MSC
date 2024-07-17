from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("./fine-tune-dreambooth", torch_dtype=torch.float16, use_safetensors=True).to("cuda")
image = pipeline("A photo of sks dog in a bucket", num_inference_steps=50, guidance_scale=7.5).images[0]
image.save("dog-bucket.png")

