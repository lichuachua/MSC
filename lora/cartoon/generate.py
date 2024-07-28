from diffusers import AutoPipelineForText2Image
import torch
import os

prompts = [
    "Create an ultra-detailed, high-resolution pixel art masterpiece of a girl with fox ears and a heart-shaped pupils. She has red hair styled in a short ponytail and a smirk on her face. She is wearing a hoodie with a hood, a fanny pack, and a denim skirt. The scene is set against a beautiful purple sunset at a beach with cinematic lighting and a cloudy sky, viewed from the left side.",
    "Illustrate an extremely detailed, high-resolution official art piece of a girl sitting outdoors. She has very long black hair and medium-sized breasts, and is looking directly at the viewer with a blush and a smile. She wears a blue dress with cleavage and bare shoulders, along with black gloves, a hat, and a skirt. The background features a blue sky with clouds and a lens flare effect. The art is in 8K resolution with an emphasis on the intricate details of her attire and the setting.",
    "Create a highly detailed 8K resolution CG artwork of a girl in a red kimono and fox mask, showcasing Japanese clothing. She has long brown hair, bangs, and wears black gloves. Her kimono features a belt and an obi, and she has double tails. Her purple eyes are visible through the mask, and she is set against a blurred background with a focus on her bare shoulders, midriff, and the intricate details of her outfit. The scene includes a lawn, sky, and buildings in the background.",
    "Draw a masterpiece of multiple girls with long hair and pink eyes, wearing school uniforms. They have various hair ornaments and clips, with some featuring bangs and a red or pink neckerchief. The artwork is indoor and focuses on the upper body of these girls, who are depicted with open mouths and engaged expressions.",
    "Create a traditional Chinese painting-style artwork of a loli character with a thin frame and medium breasts. She has red hair styled in a ponytail, a kind smile, and animal ears. Her heterochromia is evident with her red lips and fox tail. She wears earrings, frilled legwear, and a skirt, with her toes spread out."
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
