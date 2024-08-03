from diffusers import AutoPipelineForText2Image
import torch
import os

prompts = [
    "(((sitting, crossed legs))), ((Jeanne d'Arc Alter (Fate))), ((maid, maid apron, maid headdress)), best quality, looking at viewer, 1 girl, blonde hair, yellow eyes, high resolution, detailed, intricate, green background, close-up, open clothes, (beautiful face:1.15), anime style, soft lighting, indoor setting, Victorian window, (intricate details), (high quality), (illustration:1.1)",
    "anime screencap, masterpiece, best quality, 1girl, aqua eyes, baseball cap, blonde hair, closed mouth, earrings, green background, hat, hoop earrings, jewelry, looking at viewer, shirt, short hair, simple background, solo, upper body, yellow shirt",
    "solo,multicolored hair,one-hour drawing challenge,shirt,tears,hand up,thighs,cropped legs,kindergarten uniform,red eyes,blue skirt,between legs,dress shirt,kantai collection,ebifurya,looking at viewer,collared shirt,very long hair,matsuwa (kancolle),skirt,gradient hair,simple background,1girl,purple hair,sitting,white background,pleated skirt,long hair,blue shirt,name tag,white headwear,parted bangs,hand between legs,black hair,highres,blush",
    "1girl, solo, smile, indoors, blue eyes, side ponytail, classroom, black hair, breasts, sweater, looking at viewer, desk, sleeveless sweater, school desk, ribbed sweater, grin, sleeveless turtleneck, turtleneck, bare shoulders, blush, long hair, chalkboard, chair, turtleneck sweater, bangs, medium breasts, bare arms, upper body, sitting, school chair, green background",
    "mayuzumi fuyuko, 1 girl, arm support, bed sheet, black hair, black ribbon, black skirt, black thighhighs, blush, large breasts, long hair, long sleeves, looking at viewer, neck ribbon, pink shirt, ribbon, frilled shirt, sitting, skirt, smile, solo, suspender skirt, suspenders, white background, cinematic angle, cinematic lighting, anime coloring, detailed, masterpiece, best quality"
]

negative_prompts = [
    "bad-artist-anime, bad-hands-5, bad-image-v2-39000, bad_prompt_version2, bad_quality, bad-image-9600, badtth115, negprompt5, NG_DeepNegative_V1_75T, rtfclprmt315, Unspeakable-Horrors-Composition-4v,((worst quality)), low quality, (((out of frame))), (((out of borders))), ((close up)),  ((disfigured)), ((bad art)), blurry foreground, (blurry:2.0), jpeg artifacts, signature, watermark, username, blurry, artist name,  text, JPEG artifacts, signature,watermark, extra digit, fewer digits, text, error, patreon username, text font ui,  (censored), mosaic censoring, bar censor, pointless censoring, horror, cropped, (((deformed))), ((extra limbs)), (((duplicate))), ((morbid)), ((mutilated)), ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), ((ugly)),  ((bad anatomy)), (((bad proportions))), weird colors, gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck))), ((extra fingers)), missing fingers, ((mangled fingers)), glitchy,  (((((fused fingers))))), (((((too many fingers))))), (((unclear eyes))), (lowres), bad anatomy, cloned face, malformed hands, long neck, missing limb,  poorly drawn feet, disfigured, (mutated hand and finger: 1.5), (long body: 1.3), (mutation poorly drawn: 1.2), fused asshole, missing asshole, bad anus, bad pussy, bad crotch, bad crotch seam, fused anus, fused pussy, (more than 2 nipples), missing clit, bad clit, fused clit, worst face, Ugly Fingers, [thick lips], huge eyes,  multiple breasts, ((plump)),yaoi, furry, pubic hair, mosaic, (multiple moles),  ((pubic tattoo)), (deformed fingers:1.2), (long fingers:1.2), (interlocked fingers:1.2), forehead mark, facial mark, ((monochrome)),",
    "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
    "blurry, blur, watermark, logo, text, signature, nudity, nsfw, hands,",
    "nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]",
    "nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]"
]

base_dir = "./fine-tune-lora"
output_base_dir = "./output_file"
end_output_dir = "./output_file/output_file_checkpoint-10000000"

os.makedirs(end_output_dir, exist_ok=True)

try:
    pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                         torch_dtype=torch.float16).to("cuda")
    pipeline.load_lora_weights(base_dir, weight_name="pytorch_lora_weights.safetensors")
except HFValidationError as e:
    print(f"Error loading pipeline or weights: {e}")
    raise

for i, (prompt, n_prompt) in enumerate(zip(prompts, negative_prompts)):
    output_file = os.path.join(end_output_dir, f"cartoon_{i + 1}.png")
    if not os.path.exists(output_file):
        try:
            print(f"Generating image for prompt {i + 1}")
            image = pipeline(prompt=prompt, negative_prompt=n_prompt).images[0]
            image.save(output_file)
        except Exception as e:
            print(f"Error generating image for prompt {i + 1}: {e}")
    else:
        print(f"Image for prompt {i + 1} already exists, skipping.")

checkpoints = [f for f in os.listdir(base_dir)
               if os.path.isdir(os.path.join(base_dir, f)) and f.startswith('checkpoint')]

for checkpoint in checkpoints:
    checkpoint_path = os.path.join(base_dir, checkpoint)
    output_dir = os.path.join(output_base_dir, f"output_file_{checkpoint}")

    os.makedirs(output_dir, exist_ok=True)

    try:
        print(f"Processing checkpoint: {checkpoint}")
        pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                             torch_dtype=torch.float16).to("cuda")
        pipeline.load_lora_weights(checkpoint_path, weight_name="pytorch_lora_weights.safetensors")
    except HFValidationError as e:
        print(f"Error loading pipeline or weights for checkpoint {checkpoint}: {e}")
        continue

    for i, (prompt, n_prompt) in enumerate(zip(prompts, negative_prompts)):
        output_file = os.path.join(output_dir, f"cartoon_{i + 1}.png")
        if not os.path.exists(output_file):
            try:
                print(f"Generating image for prompt {i + 1} using checkpoint {checkpoint}")
                image = pipeline(prompt=prompt, negative_prompt=n_prompt).images[0]
                image.save(output_file)
            except Exception as e:
                print(f"Error generating image for prompt {i + 1} using checkpoint {checkpoint}: {e}")
        else:
            print(f"Image for prompt {i + 1} using checkpoint {checkpoint} already exists, skipping.")

print("All images generated and saved.")
