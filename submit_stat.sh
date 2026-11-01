#!/bin/bash

#SBATCH --time=0:40:00
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=30G

# copied from ~/analysis/run12/data/unfolding/good/final_0628/submit_stat.sh

PYTHON="/home/ys668/.conda/envs/my_env_2/bin/python"
source setup_cuda.sh

cd /nfs/roberts/scratch/pi_hlc7/ys668

if [ ! -d 'statistic'$1 ]
then
    mkdir 'statistic'$1
    cp ~/multifold-101326/multifold_stat.py 'statistic'$1
    cd 'statistic'$1
    echo "About to run multifold for statistics trial $1"
    export PYTHONHASHSEED=0
    srun "$PYTHON" multifold_stat.py $1
    echo "Job for statistics trial $1 is finished."
else
    echo "Skipping job for statistics trial $1"
fi