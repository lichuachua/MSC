from diffusers import AutoPipelineForText2Image
import torch
import os

prompts = [
    "Create a masterpiece portrait of a girl in 8K resolution with ultra-high detail and photorealistic quality. The girl is kneeling with her legs wide apart, in a missionary pose. The camera is positioned low and close to her, capturing her in a high-quality, ultra-detailed shot. She has streaked hair and is wearing a choker. The background features graffiti and paint splatters, with her arms on her hips, leaning back and looking directly at the camera. She also has an armband and a thigh strap, with a perfect, small, and thin face.",
    "Generate a full-body portrait of a girl with a white background and a floating petal theme. The scene is set at night with a starry sky, and the background includes flowers, a dreamcatcher, stone fragments, and beautiful detailed water. The girl has long blonde hair styled with one side up, blue eyes, and small breasts. She is wearing a white transparent babydoll, white thigh-highs, and see-through white silk stockings. The image should be in pastel colors, with detailed elements like a table, flowers, bows, and frills. The scene includes a floating island and clouds, with a focus on both character and background details.",
    "Create an ultra-illustrated, high-quality animated image of a girl standing solo in a cowboy shot. She has rabbit ears and is looking directly at the viewer. The image should feature ray tracing and global illumination for a polished, masterpiece look.",
    "Generate a high-quality image of a girl with a purple-eyed, disgusted expression. She is shown from above, wearing an orange-themed cosplay outfit that includes a collared shirt and a flared skirt. The scene is set outdoors, with the focus on her upper body, side braids, and small breasts.",
    "Create an image of a girl with long brown hair styled in twintails, with a light blush on her cheeks and a small nose blush. She is the solo focus of the image, wearing a blue bow and a hooded jacket. The scene is set outdoors at night, with a starry sky and a bench. She is sitting in a casual outfit consisting of a dark hooded jacket, a pleated skirt, and is hugging her arms. The image should capture the atmosphere of a dark night street."
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
