from diffusers import AutoPipelineForText2Image
import torch
import os

prompts = [
    "A solo girl with brown hair styled in bangs, wearing a red jacket and a white beret with a black ribbon. She has brown eyes, a blush on her cheeks, and an open mouth as she smiles and salutes with one arm behind her back. The background is simple and white, highlighting her dress and open jacket.",
    "A girl with very long brown hair styled in twintails and purple eyes. She has a light blush and fangs, wearing a blue bow and a casual hooded jacket with a pleated skirt. She is sitting on a bench outdoors at night, under a starry sky.",
    "A close-up of a girl with blue eyes, wearing a detailed dress with frilled sleeves and a brooch. Her hair is curtained and pastel-colored, with her arms behind her back. The background features a village with a bokeh effect.",
    "A girl with purple eyes and green hair styled in twintails, adorned with a hair ornament and a flower. She is wearing a purple dress with puffy short sleeves, white pantyhose, and black ribbon accents. The scene is indoors on a couch, with a window in the background, and she is looking at the viewer with a gentle blush.",
    "A girl with long hair, purple eyes, and a flower in her hair, wearing torn clothes and white thigh-highs with brown boots. She is kneeling outdoors among petals and plants, with a blush on her cheeks and an open mouth. Her outfit includes detached sleeves and a dress with frills, and she is surrounded by a red flower and other details like animal ears."
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