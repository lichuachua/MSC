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
### 准备
1. 下载安装diffusers包
2. 运行'cd diffusers'
3. 运行'pip install .'
4. 运行'cd examples/dreambooth'
5. 运行'pip install -r requirements.txt'
### 训练
6. 回到当前文件夹，运行'qsub train.sh'，等待当前任务完成
### 生成
7. 任务完成，在当前文件夹运行'qsub generate.sh'

