# single core submission script

#Run with current environment (-V) and in the current directory (-cwd)
#$ -V -cwd

#Request some time- min 15 mins - max 48 hours
#$ -l h_rt=00:15:00

#Request some memory per core
#$ -l h_vmem=1G

#Get email at start and end of the job
#$ -m be

#$ -l coproc_v100=3

#Now run the job
module load cuda
nvidia-smi


python train_dreambooth.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"  \
  --instance_data_dir="./dog-examples" \
  --output_dir="./fine-tune-dreambooth" \
  --instance_prompt="a photo of a sks dog" \
  --resolution=128 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400
