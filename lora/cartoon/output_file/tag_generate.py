from diffusers import AutoPipelineForText2Image
import torch
import os

prompts = [
    "katou megumi, 1girl, solo, bangs, brown hair, hat, white headwear, dress, white background, looking at viewer, simple background, brown eyes, open mouth, short hair, ribbon, jacket, red jacket, black ribbon, blush, open clothes, smile, beret, salute, open jacket, arm behind back, collarbone, neck ribbon, long sleeves, :d, upper body, <lora:Misaki Kurehito_XL:0.8>",
    "1girl, purple eyes, blush, long hair, twintails,  solo focus, brown hair, bow, bangs, hair between eyes, blue bow, nose blush,  light blush, :o, fang,  very long hair,  breasts,  blonde hair,  fangs,  outdoor, night, street, starry sky, bench, sitting., hooded jacket, casual,dark night ,night sky, arm hug, open jacket, pleated skirt",
    "masterpiece, best quality, detailed, 1girl, close up,curtained hair, blue eyes, brooch, laces, frilled sleeves, dress, village, bokeh, arms behind back,pastel colorstheme,",
    "masterpiece,best quality,StyleD01,1girl, solo, purple eyes, dress, short sleeves, green hair, pantyhose, twintails, hair ornament, long hair, hair flower, looking at viewer, flower, sitting, blush, ribbon, puffy sleeves, bangs, couch, indoors, purple dress, puffy short sleeves, closed mouth, frills, hair ribbon, white pantyhose, wrist cuffs, black ribbon, frilled dress, hair between eyes, window, on couch, breasts, hair intakes, feet out of frame, neck ribbon, choker, bow",
    "masterpiece,best quality,StyleD01,1girl, solo, thighhighs, long hair, flower, boots, breasts, hair ornament, looking at viewer, detached sleeves, hair flower, blush, torn clothes, white thighhighs, brown footwear, dress, cleavage, purple eyes, bangs, kneeling, hair between eyes, red flower, frills, all fours, petals, plant, medium breasts, white flower, closed mouth, bare shoulders, long sleeves, animal ears, outdoors, knee boots",
]

base_dir = "./fine-tune-lora"
output_base_dir = "./output_file"

end_output_dir = "./output_file/output_file_checkpoint-10000000"

os.makedirs(end_output_dir, exist_ok=True)

pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                     torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights(base_dir, weight_name="pytorch_lora_weights.safetensors")

for i, prompt in enumerate(prompts):
    output_file = os.path.join(end_output_dir, f"cartoon_{i + 1}.png")
    if not os.path.exists(output_file):
        print(f"Generating image for prompt {i + 1}")
        image = pipeline(prompt=prompt).images[0]
        image.save(output_file)
    else:
        print(f"Image for prompt {i + 1} already exists, skipping.")

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
        output_file = os.path.join(output_dir, f"cartoon_{i + 1}.png")
        if not os.path.exists(output_file):
            print(f"Generating image for prompt {i + 1} using checkpoint {checkpoint}")
            image = pipeline(prompt=prompt).images[0]
            image.save(output_file)
        else:
            print(f"Image for prompt {i + 1} using checkpoint {checkpoint} already exists, skipping.")

print("All images generated and saved.")