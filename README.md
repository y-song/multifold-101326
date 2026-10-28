# multifold-101326

- Environment
  - [my_env_2.yml](https://github.com/y-song/multifold-101326/blob/master/my_env_2.yml)
  - [setup_cuda.sh](https://github.com/y-song/multifold-101326/blob/master/setup_cuda.sh)

- Source code change
  - Changed `~/.conda/envs/my_env_2/lib/python3.10/site-packages/energyflow/archs/archbase.py` to [archbase.py](https://github.com/y-song/multifold-101326/blob/master/archbase.py)

- Test MultiFold run
  - [multifold.py](https://github.com/y-song/multifold-101326/blob/master/multifold.py)
  ```
  salloc --time 4:0:00 --gpus 1 --partition gpu_devel --mem 100G
  module load miniconda
  conda activate my_env_2
  source setup_cuda.sh
  python multifold.py 1  
  ```
  Output is saved as csv files at `~/scratch_pi_hlc7/ys668/test_101326/`

- Test plotting script
  - [plot.ipynb](https://github.com/y-song/multifold-101326/blob/master/plot.ipynb) reads in MultiFold output and plots results

- Batch job run
  - [submit.sh](https://github.com/y-song/multifold-101326/blob/master/submit.sh)
  - [submit_all.sh](https://github.com/y-song/multifold-101326/blob/master/submit_all.sh)
  ```
  module load miniconda
  conda activate my_env_2
  sbatch submit_all.sh
  ```  