#!/bin/bash

# copied from ~/analysis/run12/data/unfolding/good/final_0628/submit_all.sh

# for i in {1..100}
for i in 1 
do
	sbatch submit.sh ${i}
done
