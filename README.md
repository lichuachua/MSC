# 1.blog-note
## Introduce:
  - The basic usage of Stable Diffusion for generating some instance images.
## Structure
  - blog-nate.py - Source file  
  - blog-note.sh - Script file  
  - output_image - Output file (generated image)
## Running steps
 - Run 'qsub blog-note.sh' in the current directory

# 2.dreamboothdreambooth
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
1. Download and install the diffusers package
2. Run 'cd diffusers'
3. Run 'pip install .'
4. Run 'cd examples/dreambooth'
5. Run 'pip install -r requirements.txt'
### Training
6. Return to the current folder, run 'qsub train. sh', and wait for the current task to complete
### Generate
7. Task completed, run 'qsub generate. sh' in the current folder

