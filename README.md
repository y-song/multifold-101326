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
  - Input embedding file: `/nfs/roberts/home/ys668/output/run12/all050124_rc_notrueptcons.root`  
    Input data file: `/nfs/roberts/home/ys668/output/run12/data062824.root`  
    Test run output directory: `/nfs/roberts/scratch/pi_hlc7/ys668/test_101326/`

- Test plotting script
  - [plot.ipynb](https://github.com/y-song/multifold-101326/blob/master/plot.ipynb) reads in MultiFold output and plots results

- Batch job run
  - [submit.sh](https://github.com/y-song/multifold-101326/blob/master/submit.sh)
  - [submit_stat.sh](https://github.com/y-song/multifold-101326/blob/master/submit_stat.sh)
  - [submit_all.sh](https://github.com/y-song/multifold-101326/blob/master/submit_all.sh)
  ```
  module load miniconda
  conda activate my_env_2
  sbatch submit_all.sh
  ```  

- Nominal MultiFold run
  - This is already done.
  - [multifold.py](https://github.com/y-song/multifold-101326/blob/master/multifold.py)
  - Input embedding file: `/nfs/roberts/home/ys668/output/run12/all050124_rc_notrueptcons.root`  
    Output directory: `/nfs/roberts/project/pi_hlc7/ys668/run12/csvs/final_0628/20235003_data_batch50000_notrueptcons/`

- Runs for systematic uncertainties
  - Detector effect variations:
    - This is already done.
    - What was done: Modified [multifold.py](https://github.com/y-song/multifold-101326/blob/master/multifold.py). Changed input embedding file (line 119) and output path (line 204). No need to change input data file (line 128). In principle, also need to change the fake rates (line 134), but the variations we consider have minimal effect on them (same numerical values within two decimals).  
    - Input embedding files: `/nfs/roberts/home/ys668/output/run12/all050124_rc_eff096_notrueptcons.root`, `/nfs/roberts/home/ys668/output/run12/all050124_rc_tow104_notrueptcons.root`, `/nfs/roberts/home/ys668/output/run12/all050124_rc_hadcorr05_notrueptcons.root`  
    Output directories: `/nfs/roberts/pi/pi_lbh35/youqi/csvs/20235003_eff096_notrueptcons/`, `/nfs/roberts/pi/pi_lbh35/youqi/csvs/20235003_tow104_notrueptcons/`, `/nfs/roberts/project/pi_hlc7/ys668/run12/csvs/final_0628/20235003_hadcorr05_notrueptcons/`
  - Prior variations:
    - This hasn't been done.
    - What needs to be done: Modify [reweight_pythia_to_pythia8.ipynb](https://github.com/y-song/multifold-101326/blob/master/reweight_pythia_to_pythia8.ipynb). This reads in self-generated pythia8 jets `/nfs/roberts/pi/pi_lbh35/youqi/detroit_tune/pythia_*pthat*_multifold_truth.txt`, reweights MATCHED pythia6 jets by TRUTH pythia8/pythi6 ratio, and saves the output pythia6 jets into a csv file. Then [multifold_pythia8.py](https://github.com/y-song/multifold-101326/blob/master/multifold_pythia8.py) reads in the csv to run MultiFold. Similar for herwig, to be done with [reweight_pythia_to_herwig.ipynb](https://github.com/y-song/multifold-101326/blob/master/reweight_pythia_to_herwig.ipynb) and [multifold_herwig.py](https://github.com/y-song/multifold-101326/blob/master/multifold_herwig.py).

- Runs for statistical bootstrap
  - Data resampling:
    - This is already done.
    - [multifold_stat.py](https://github.com/y-song/multifold-101326/blob/master/multifold_stat.py)  
    - Output direcotry: `/nfs/roberts/pi/pi_lbh35/youqi/csvs/20235003_datastat_notrueptcons/`
  - Embedding resampling:
    - This is already done.
    - [multifold_mc_stat.py](https://github.com/y-song/multifold-101326/blob/master/multifold_mc_stat.py)  
    - Output directory: `/nfs/roberts/pi/pi_lbh35/youqi/csvs/20235003_stat_notrueptcons/`

- Full plotting script
  - [plot_full.ipynb](https://github.com/y-song/multifold-101326/blob/master/plot_full.ipynb)  

- Potential improvement
  - Currently, `multifold.py` saves the unfolded weights as separate CSV files for each iteration and seed, which takes up a lot of disk space. One possible improvement is to save the weights for all iterations of each seed in a single compressed NumPy (.npz) file instead. This would also require a small change to the `# Read in unfolded` section of `plot.ipynb` to load the .npz files instead of CSVs. The rest of the plotting code could remain unchanged.