from diffusers import AutoPipelineForText2Image
import torch
import os

prompts = [
    "Create an image of Yunaka from Fire Emblem Engage. She has long red hair, red eyes, and a playful expression with one eye closed. She wears a black bodysuit with bare shoulders and a flowing black cape draped behind her. Her hair has a star-shaped ornament, and she has a gentle blush on her cheeks. The background should be white.",
    "Draw Kasashi (Kasasi008) from Kantai Collection in an alternate costume. She has long pink hair and pink eyes. She wears a ribbed black turtleneck and has a confident expression with a blush on her cheeks. Her glasses are off, and the background is a grey gradient.",
    "Illustrate Nakasu Kasumi from Love Live! Nijigasaki with short light brown hair, red eyes, and a school uniform. She has a crescent and star hair ornament and a gentle blush. She is smiling brightly and looking directly at the viewer. The background should be simple.",
    "Sketch Osaki Tenka from The Idolmaster Shiny Colors. She has long brown hair with swept bangs, yellow eyes, and a gentle blush. She wears a school uniform with a sweater and a necktie. She has a slightly open mouth and a charming pose. The background is white.",
    "Create an image of Anya from Spy x Family with long white hair styled in a side ponytail and a pink bow with a rabbit ornament. She wears an open white jacket over a white shirt, white shorts, and red socks. One shoe is removed, and she is holding it. She has medium breasts, a thigh strap, and a playful expression. The background should be simple."
]

base_dir = "./fine-tune-lora"
output_base_dir = "./output_file"

end_output_dir = "./output_file/output_file_checkpoint-10000000"

os.makedirs(end_output_dir, exist_ok=True)

pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                     torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights(base_dir, weight_name="pytorch_lora_weights.safetensors")

for i, prompt in enumerate(prompts):
    print(f"Generating image for prompt {i + 1}")
    image = pipeline(prompt).images[0]
    image.save(os.path.join(end_output_dir, f"cartoon_{i + 1}.png"))

checkpoints = [f for f in os.listdir(base_dir)
               if os.path.isdir(os.path.join(base_dir, f)) and f.startswith('checkpoint')]

for checkpoint in checkpoints:
    checkpoint_path = os.path.join(base_dir, checkpoint)
    output_dir = os.path.join(output_base_dir, f"output_file_{checkpoint}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing checkpoint: {checkpoint}")
    pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                         torch_dtype=torch.float16).to("cuda")
    pipeline.load_lora_weights(checkpoint_path, weight_name="pytorch_lora_weights.safetensors")

    for i, prompt in enumerate(prompts):
        print(f"Generating image for prompt {i + 1} using checkpoint {checkpoint}")
        image = pipeline(prompt).images[0]
        image.save(os.path.join(output_dir, f"cartoon_{i + 1}.png"))

print("All images generated and saved.")
