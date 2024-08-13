# 1.blog-note

## Introduce:

- The basic usage of Stable Diffusion for generating some instance images.

## Structure

- blog-nate.py - Source file
- blog-note.sh - Script file
- output_image - Output file (generated image)

## Running steps

- Run 'qsub blog-note.sh' in the current directory

# 2.dreambooth

## Introduce:

- Fine tune the stable diffusion model using dreambooth,

## Structure

- dog-examples - fine-tuning the dataset used
- train.sh - Script for training the model
- fine-tune-dreambooth - a new model after fine-tuning
- generate.py - Source file for generate images
- generate.sh - Footstep file for generate images

## Running steps

### Prepare

1. Enter the dreambooth folder, Download and install the diffusers package - run 'git
   clone https://github.com/huggingface/diffusers'
2. Run 'cd diffusers'
3. Run 'pip install .'
4. Run 'cd examples/dreambooth'
5. Run 'pip install -r requirements.txt'

### Training

6. Return to the current folder(dreambooth folder), run 'qsub train.sh', and wait for the current task to complete
   After the training is completed, you will see the trained model file in the 'fine-tune-dreambooth' folder, as shown
   below：
   <img width="1014" alt="image" src="https://github.com/user-attachments/assets/2b6a8d2f-4d1d-42fd-a37b-c1d05d05189e">

### Generate

7. Task completed, run 'qsub generate.sh' in the current folder

# 3.lora

## Introduce:

- Fine tune the stable diffusion model using Lora,

## Structure

- pokemon
    - pokemon_dataset - fine-tuning the dataset used lora about pokemon
    - train.sh - Script for training the model
    - fine-tune-lora - a new model after fine-tuning
    - generate.py - Source file for generate images
    - generate.sh - Footstep file for generate images

- cartoon
    - cartoon_dataset - fine-tuning the dataset used lora about cartoon 800 image
    - train.sh - Script for training the model
    - fine-tune-lora - a new model after fine-tuning
    - generate.py - Source file for generate images
    - generate.sh - Footstep file for generate images

- cartoon_new
    - cartoon_dataset - fine-tuning the dataset used lora about cartoon 80 high-qulity image
    - train.sh - Script for training the model
    - fine-tune-lora - a new model after fine-tuning
    - generate.py - Source file for generate images
    - generate.sh - Footstep file for generate images

- sd_xl_cartoon_new
    - cartoon_dataset - fine-tuning the dataset used lora and stable_diffusion_xl about cartoon 80 high-qulity image
    - train.sh - Script for training the model
    - fine-tune-lora - a new model after fine-tuning
    - generate.py - Source file for generate images
    - generate.sh - Footstep file for generate images

## Running steps

### Prepare

1. Enter the lora folder, Download and install the diffusers package - run 'git
   clone https://github.com/huggingface/diffusers'
2. Run 'cd diffusers'
3. Run 'pip install .'
4. Run 'cd examples/text_to_image'
5. Run 'pip install -r requirements.txt'

### Training

6. Return to the current folder(lora/pokemon folder), run 'qsub train.sh', and wait for the current task to complete
   After the training is completed, you will see the trained model file in the 'fine-tune-lora' folder, as shown
   below：
   ![img](https://github.com/user-attachments/assets/c9c67f19-70ba-41d5-ab10-10644948b280)

### Generate

7. Task completed, run 'qsub generate.sh' in the current folder

# 4. Tool

## Annotation

### address: tools/tag-generate.ipynb

- DeepDanbooru
- BLIP

### address: https://github.com/lichuachua/wd14-tag

- WD14

# 5. Clip-score

## address: tools/clip-score.py

# 5. Questionnaire

## address: tools/questionnaire.ipynb

# 6. Report Images Generate

## address: tools/cartoon-new-generate.ipynb
