from diffusers import AutoPipelineForText2Image
import torch
import os

prompts = [
    "(masterpiece) 1girl,portrait,(8k, RAW photo, best quality, ultra high res, photorealistic, ultra-detailed),legs wide,kneeling,missionary,camera low,camera close,best quality,1girl,solo,streaked hair,choker,(graffiti:1.25),paint splatter,arms on hips,leaning back,looking at camera,armband,thigh strap,streaked hair,head down,head tilt,tight,thin,small,perfect face,ninym ralei,",
    "(no shoe:1.2),[(white background:1.5)::5],(bottle bottom:0.9),mid shot,full body,night,night sky,starry night,flowers,(dreamcatcher:1.3),floating petal,stone fragment,beautiful detailed water,cloud,sun,dusk,sunset,,(solo:1.2),(Masterpiece),(Best quality),(Extremely�Detailed�CG�Unity�8k�Wallpaper),(high resolution),(Best background details:1.26),(Best character details:1.36),(pastel (medium):1.35),sparkle,1girl,(full body),(loli:1.2),(kawaii:1.12),(full shot:1.2),(pastel \(medium\):1.16),long blonde hair,one side up,blue eyes,small breasts,white transparent babydoll,(white thighhighs:1.1)+(see-through:1.1),white silk stocking,illustration,Accumulate,table,flower,bow,bangs,tree,frills,bowtie,pink flower,closed mouth,shirt,outdoors,pink bow,grass,yellow bow,saucer,[(floating island:1.2,cloud)::5]",
    "((((ultra illustrated style:1.0)))),best quality,best animated,masterpiece,ray tracing, global illumination,1girl, solo,cowboy shot,looking at viewer, standing,rabbit ears",
    "masterpiece,best quality,upper body,1girl,collared_shirt and flared_skirt as material1,orange theme,cosplay,outside border,side braids,small breasts,purple eyes,disgust,from above",
    "1girl, purple eyes, blush, long hair, twintails,  solo focus, brown hair, bow, bangs, hair between eyes, blue bow, nose blush,  light blush, :o, fang,  very long hair,  breasts,  blonde hair,  fangs,  outdoor, night, street, starry sky, bench, sitting., hooded jacket, casual,dark night ,night sky, arm hug, open jacket, pleated skirt, ."
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
