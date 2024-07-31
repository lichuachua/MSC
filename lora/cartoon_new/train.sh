# single core submission script

#Run with current environment (-V) and in the current directory (-cwd)
#$ -V -cwd

#Request some time- min 15 mins - max 48 hours
#$ -l h_rt=40:00:00

#Request some memory per core
#$ -l h_vmem=2G

#Get email at start and end of the job
#$ -m be

#$ -l coproc_v100=1

#Now run the job
module load cuda
nvidia-smi


python ../diffusers/examples/text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"  \
  --dataset_name="./cartoon_dataset" \
  --dataloader_num_workers=8 \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir="./fine-tune-lora" \
  --hub_model_id="cartoon-new-lora" \
  --checkpointing_steps=200 \
  --validation_prompt="anime screencap, masterpiece, best quality, 1girl, aqua eyes, baseball cap, blonde hair, closed mouth, earrings, green background, hat, hoop earrings, jewelry, looking at viewer, shirt, short hair, simple background, solo, upper body, yellow shirt" \
  --seed=1337
