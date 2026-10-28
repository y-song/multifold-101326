#!/bin/bash

#SBATCH --time=1:0:00
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=30G

# copied from ~/analysis/run12/data/unfolding/good/final_0628/submit.sh

PYTHON="/home/ys668/.conda/envs/my_env_2/bin/python"
source setup_cuda.sh

cd /nfs/roberts/scratch/pi_hlc7/ys668

if [ ! -d 'seed'$1 ]
then
	mkdir 'seed'$1
	cp ~/multifold-101326/multifold.py 'seed'$1
	cd 'seed'$1
	srun "$PYTHON" multifold.py $1
	echo "Job for seed $1 is finished."
else
	echo "Skipping job for seed $1"
fi