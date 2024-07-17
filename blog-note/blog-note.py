#!/usr/bin/env python
# coding: utf-8

# Original article: https://huggingface.co/blog/stable_diffusion

# Install diffusers
# Install scipy, ftfy, transformers, and accelerate for faster loading

# In[1]:

# get_ipython().system('pip install diffusers')
# get_ipython().system('pip install transformers scipy ftfy accelerate')

# Load the model

# In[2]:

from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

# Generate an image (each run will give you a different image)

# In[3]:

pipe = pipe.to("cuda")

prompt = "a photograph of an astronaut riding a horse"
image = pipe(prompt).images[0]

image.save("./output_image/astronaut_rides_horse.png")
image

# If at any point you get a black image, it may be because the model's built-in content filter detected NSFW content.
# NSFW means "Not Safe For Work", indicating content not suitable for workplace viewing, such as nudity, pornography, or violence.
# Check if the prompt contains NSFW information.

# In[4]:

result = pipe(prompt)
print(result)
prompt_nsfw = "a beautiful and sexy girl"
result = pipe(prompt_nsfw)
print(result)

# To ensure deterministic output, you can seed the random number generator and pass it to the pipeline.
# Each time you use the generator with the same seed, you will get the same image output.

# In[5]:

import torch

# Create a random number generator on the GPU and set the seed to 1024
generator = torch.Generator("cuda").manual_seed(1024)
image = pipe(prompt, guidance_scale=7.5, generator=generator).images[0]

image.save("./output_image/astronaut_rides_horse_generator.png")
image

# Generally, the more steps you use, the better the result, but the longer it takes to generate.
# Stable Diffusion performs well with relatively few steps, so we recommend using the default number of inference steps.
# If you want faster results, use a smaller number. If you want higher quality results, use a larger number.
# Let's try running the pipeline with fewer denoising steps.

# In[6]:

import torch

generator = torch.Generator("cuda").manual_seed(1024)
image = pipe(prompt, guidance_scale=7.5, num_inference_steps=15, generator=generator).images[0]

image.save("./output_image/astronaut_rides_horse_num_inference_steps.png")
image

# Using only 15 denoising steps significantly reduces the quality of the generated result.
# As mentioned, a sufficient number of denoising steps is typically needed to generate high-quality images.

# guidance_scale is known as classifier-free guidance, which simply forces the generator to better match the prompt,
# potentially at the expense of image quality or diversity. Values between 7.5 and 8 are usually good choices for Stable Diffusion.
# By default, the pipeline uses 7.5. If you use very large values, the images may look good but diversity will be reduced.

# Next, let's see how to generate multiple images of the same prompt at once.
# First, we'll create an image_grid function to help visualize them nicely in a grid.

# In[7]:

from PIL import Image

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

# We can generate multiple images of the same prompt by simply using a list that repeats the same prompt multiple times.
# We will send the list to the pipeline.

# In[8]:

num_images = 3
prompt = ["a photograph of an astronaut riding a horse"] * num_images

images = pipe(prompt).images

image = image_grid(images, rows=1, cols=3)

image.save("./output_image/astronaut_rides_horse_several.png")
image

# By default, Stable Diffusion generates 512x512 pixel images.
# You can easily override the defaults using the height and width parameters to create rectangular images with portrait or landscape orientation.
# When choosing image dimensions, we recommend the following:
# Ensure both height and width are multiples of 8.
# Below 512 may result in lower image quality.
# Exceeding 512 in both dimensions may cause repeated image areas (loss of global coherence).
# The best way to create non-square images is to use 512 in one dimension and a value greater than that in the other.

# In[9]:

prompt = "a photograph of an astronaut riding a horse"
image = pipe(prompt, height=512, width=768).images[0]
image.save("./output_image/astronaut_rides_horse_customer.png")
image
