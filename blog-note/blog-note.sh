#R single core submission script

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
python blog-note.py

