#!/bin/bash
###
# job name
#SBATCH --job-name=deeport
# maximum job time in D-HH:MM
#SBATCH --time=00:48:00
# job stderr file
#SBATCH --error=<PATH to dumping directory DeepOrientation/data/polnet/res/logs/deeport_errors_%j.err>
# job stdout file
#SBATCH --output=<PATH to dumping directory DeepOrientation/data/polnet/res/logs/deeport_output_%j.out>
# Specify the GPU partition
#SBATCH --partition=gpu
# Specify how many GPUs we would like to use
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
###
data="polnet"
name=$SLURM_JOB_ID
logpath="< PATH to jobid directory in DeepOrientation/data/$data/res/logs/$name>"
mkdir -p $logpath

# Load Anaconda and activate our environment with Tensorflow installed
export PATH="<PATH to miniconda3/bin:$PATH>"
export PATH="<PATH to cuda-12.0/bin:$PATH>"
export LD_LIBRARY_PATH="<PATH to cuda-12.0/lib64:$LD_LIBRARY_PATH>"
#suorce <initiating miniconda3/bin/conda init bash>
source ~/.bashrc
conda activate tfenv

python train_slurm.py $SLURM_JOB_ID
conda deactivate


