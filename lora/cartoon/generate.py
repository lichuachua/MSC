from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights("./fine-tune-lora", weight_name="pytorch_lora_weights.safetensors")
image = pipeline("A girl in a red dress standing on a wooden fence").images[0]
image.save("cartoon.png")
