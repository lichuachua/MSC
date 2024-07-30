from diffusers import AutoPipelineForText2Image
import torch
import os

prompts = [
    "katou megumi, 1girl, solo, bangs, brown hair, hat, white headwear, dress, white background, looking at viewer, simple background, brown eyes, open mouth, short hair, ribbon, jacket, red jacket, black ribbon, blush, open clothes, smile, beret, salute, open jacket, arm behind back, collarbone, neck ribbon, long sleeves, :d, upper body, <lora:Misaki Kurehito_XL:0.8>",
    "best quality,high resolution,distinct image,best quality,high resolution,distinct image,Original Characters,Natural Volumetric Lighting And Best Shadows,Deep Depth Of Field,Sharp Focus,Portrait Of Stunningly Beautiful Petite Girl,Soft Delicate Beautiful Attractive Face With Alluring Yellow Eyes,Lovely Small Breasts,Sharp Eyeliner,Seductive Smiling,Open Mouth With Cute Fangs Out,Windswept Disheveled Brown Hair,Thick Layered Medium Twintail Hairstyles,Blush Eyeshadow With Thick Eyelashes,Parted Lips,Applejack Hat,Oversized Pop-Art Jacket,Slim Waist With Open Cute Navel,Denim Jeans Pants With Buckle Belt,(Messy Painted Body:1.05),(Holding Spray Paint Can:1.1),(Graffiti Murals Wall Background:1.15),(Standing On Narrow City Streets Crossword:1.2),(Highest Quality, Amazing Details:1.4),Masterpiece,Bloom,Picturesque,Brilliant Colorful Paintings,masterpiece,Hd",
    "science fiction,robot,solo,cable,mecha,no humans,humanoid robot,white background,standing,looking ahead,helmet,grey background,1other,wire,airport,cowboy_shot,eye-contact,depth of field,cinematic_angle,moody lighting,Cinematic Lighting,comic,8 Bit Game,illustration,highres,fantasy,ban,paleturquoise,lightgray,darkorange,orange",
    "huowu,1girl, solo, black hair, long hair,white thighhighs, chair, high heels, cosplay,((upper body)),young and beautiful, tall and beautiful, fair face, perfect features, tall and beautiful, fair skin, good figure, 4k,cg,.holding fan, blue theme,gameCG background,3Drender,realistic,cinematic lighting,smooth fog,detailed face, <lora:chilloutmixss_xss10:0.1> <lora:japaneseDollLikeness_v10:0.3>light smile, (((The Architectural Background of Japanese Style))).",
    "1girl, purple eyes, blush, long hair, twintails,  solo focus, brown hair, bow, bangs, hair between eyes, blue bow, nose blush,  light blush, :o, fang,  very long hair,  breasts,  blonde hair,  fangs,  outdoor, night, street, starry sky, bench, sitting., hooded jacket, casual,dark night ,night sky, arm hug, open jacket, pleated skirt",
    "masterpiece, best quality, 1girl, smile, cat ears, long pink hair, arms behind back, close-up, bokeh, (blurry background:1.1),",
    "masterpiece, best quality, 1girl, up-close, from side, hairclip, smile, white hair, red eyes, looking at viewer, blue eyes, hands in pockets, up close, wearing a yellow rain jacket and dark pants, rainy day, long hair, walking in a country road at sunset,",
    "masterpiece, best quality, 1girl, up-close, from side, hairclip, smile, white hair, red eyes, looking at viewer, blue eyes, hands in pockets, up close, wearing a yellow rain jacket and dark pants, rainy day, long hair, walking in a country road at sunset,",
    "masterpiece, best quality, detailed, 1girl, close up,curtained hair, blue eyes, brooch, laces, frilled sleeves, dress, village, bokeh, arms behind back,pastel colorstheme,",
    "masterpiece,best quality,StyleD01,1girl, solo, purple eyes, dress, short sleeves, green hair, pantyhose, twintails, hair ornament, long hair, hair flower, looking at viewer, flower, sitting, blush, ribbon, puffy sleeves, bangs, couch, indoors, purple dress, puffy short sleeves, closed mouth, frills, hair ribbon, white pantyhose, wrist cuffs, black ribbon, frilled dress, hair between eyes, window, on couch, breasts, hair intakes, feet out of frame, neck ribbon, choker, bow",
    "masterpiece,best quality,StyleD01,1girl, solo, thighhighs, long hair, flower, boots, breasts, hair ornament, looking at viewer, detached sleeves, hair flower, blush, torn clothes, white thighhighs, brown footwear, dress, cleavage, purple eyes, bangs, kneeling, hair between eyes, red flower, frills, all fours, petals, plant, medium breasts, white flower, closed mouth, bare shoulders, long sleeves, animal ears, outdoors, knee boots",
    "best quality,best animated,masterpiece,ray tracing, global illumination,1girl, solo, breasts, long hair, underwear,school uniform, outdoors,"
]

base_dir = "./fine-tune-lora"
output_base_dir = "./output_file"

end_output_dir = "./output_file/output_file_checkpoint-10000000"

os.makedirs(end_output_dir, exist_ok=True)

pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
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
    pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                         torch_dtype=torch.float16).to("cuda")
    pipeline.load_lora_weights(checkpoint_path, weight_name="pytorch_lora_weights.safetensors")

    for i, prompt in enumerate(prompts):
        print(f"Generating image for prompt {i + 1} using checkpoint {checkpoint}")
        image = pipeline(prompt).images[0]
        image.save(os.path.join(output_dir, f"cartoon_{i + 1}.png"))

print("All images generated and saved.")
