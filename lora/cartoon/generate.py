from diffusers import AutoPipelineForText2Image
import torch
import os

prompts = [
    "best quality, ultra-detailed, masterpiece, hires, 8k,stand up, pixel art, girl, loli,thin,short ponytail,red hair,smirk,fox ears,heart-shaped pupils,tail, hood,hoodie,fanny pack,denim skirt,denim skirt, beautiful purple sunset at beach, cinematic lighting,cloudy, view of left side",
    "masterpiece,official art,extremely detailed cg unity 8k wallpaper,highly detailed,absurdres,8k resolution,1girl,solo,long hair,breasts,looking at viewer,blush,smile,bangs,skirt,black hair,gloves,hat,dress,cleavage,bare shoulders,medium breasts,sitting,very long hair,flower,pantyhose,outdoors,parted lips,sky,sleeveless,day,black gloves,cloud,hand up,black skirt,official alternate costume,blue sky,parted bangs,grey eyes,black pantyhose,bare arms,feet out of frame,blue dress,lens flare,blue headwear,brown pantyhose,purple flower",
    "Red kimono,fox mask,long hair,Japanese clothing,choker,black gloves,bangs,brown hair,gloves,bare shoulders,kimono,shut up,belt,mask,double tails,purple eyes,obi,sandals,head mask,masterpiece,official art,extremely detailed CG uniform 8K wallpaper,highly detailed,absurd,8K resolution,1girl,solo,long hair,breasts,looking at viewer,gloves,blurry background,navel,bare shoulders,jacket,ponytail,grey hair,shorts,black gloves,midriff,belt,fingerless gloves,black jacket,crop top,hand on hip,grey eyes,v,drill hair,black shorts,sunglasses,bandaid,eyewear on head,bubble blowing,chewing gum,bronya zaychik,sliverwolf,outdoors,lawn,sky,buildings",
    "1girls,bangs,blue_neckerchief,hair_ornament,hairband,hairclip,long_hair,long_sleeves,multiple_girls,neckerchief,open_mouth,pink_eyes,pink_hair,pink_neckerchief,red_neckerchief,school_uniform,serafuku,shirt,star_\(symbol\),upper_body,white_shirt,indoor,masterpiece,best quality,",
    "anime,Traditional Chinese painting,loli,=thin,medium breasts,ponytail,red hair,kind smile,animal ears,heterochromia,red lip,fox tail,earrings,frilled legwear,earrings,skirt,spread toes"
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
