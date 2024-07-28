from diffusers import AutoPipelineForText2Image
import torch
import os

prompts = [
    "Create an image of Yunaka from Fire Emblem Engage, featuring a young woman with long red hair and red eyes. She should be wearing a black bodysuit with bare shoulders and a flowing black cape. The cape should be draped elegantly behind her. Her hair is adorned with a star-shaped ornament, and she has a gentle blush on her cheeks. Her expression should be a mix of a smile and a playful look, with one eye closed and her mouth open. The background should be white to highlight her vibrant appearance.",
    "Create an image of a character from Kantai Collection in an alternate costume. The character should be Kasashi (Kasasi008), depicted with long pink hair and pink eyes. She is wearing a ribbed black turtleneck shirt that hugs her figure, with a grey gradient background. Her glasses are removed and not visible, revealing her eyes clearly. The character should have a blush on her cheeks and a confident expression. The focus should be on her upper body, emphasizing her large breasts and the details of her attire. Her hair should be styled to fall between her eyes.",
    "Create an image of Nakasu Kasumi from Love Live! Nijigasaki High School Idol Club. She should be depicted with short light brown hair, adorned with a crescent and star-shaped hair ornament. Her hair should have bangs and sidelocks. She has red eyes and a gentle blush on her cheeks. She is wearing the Nijigasaki Academy school uniform. Her expression should be a bright smile, and she should be looking directly at the viewer. The character should be shown solo, with a focus on her small breasts and overall cheerful appearance.",
    "Create a sketch of Osaki Tenka from The Idolmaster Shiny Colors. She should have long brown hair with swept bangs and a necktie. Her hair should have a natural flow, and she should be depicted with yellow eyes and a gentle blush on her cheeks. She is wearing a school uniform with a sweater that has sleeves extending past her wrists. Her mouth should be slightly open, and she should be looking directly at the viewer. The background should be simple and white, focusing on her upper body and capturing a charming and expressive pose.",
    "Create an image of Anya from Spy x Family with a playful and detailed appearance. She should have long white hair styled with a side ponytail and a pink bow with a rabbit hair ornament. Her hair should fall between her eyes, and she should have a red-eyed gaze. She is wearing an open white jacket with long sleeves and a white shirt underneath, paired with white shorts and red socks. One shoe is removed, and she is holding it along with another shoe, revealing white footwear. She has medium breasts and a thigh strap, and she should be shown holding a box. The background should be simple and complementary, focusing on the details of her outfit and accessories."
]

base_dir = "./fine-tune-lora"
output_base_dir = "./output_file"

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

