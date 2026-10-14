# Important package versions: previous and current MultiFold environments

Both GitHub YAML files were fetched again on October 13, 2026 (Pacific time). The updated current YAML includes the pip NVIDIA CUDA packages installed for TensorFlow GPU support. Package versions below come from those files; shell runtime settings are described separately.

### Notes from human

- [Previous environment.yml](https://github.com/y-song/multid_jet_substructure/blob/main/multifold/environment.yml), used for running MultiFold on Grace. In principle I could have just recreated this environment on Bouchet (and then additionally installed relevant packages for jupyterlab, see below), but for some reason only the conda installed packages were successfully recreated; the pip installed packages didn't get re-installed. This led to (a painful process of) re-installation of newer versions of TensorFlow, CUDA, TensorRT, EnergyFlow, and tf-keras, some of which were not compatible with basic packages such as NumPy. The result is:
- [Current my_env_2.yml](https://github.com/y-song/multifold-101326/blob/master/my_env_2.yml), used for running MultiFold on Bouchet.
- Pip install commands and documentation links because I already have them saved:
  - `python -m pip install tensorflow[and-cuda]`. See [documentation](https://www.tensorflow.org/install/pip) for more information.
  - `python -m pip install --upgrade tensorrt`. See [documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/install-pip.html#) for more information.
  - `pip install energyflow`. See [documentation](https://energyflow.network/installation/#install-via-pip) for more information.
  - `pip install tf-keras`. See [documentation](https://pypi.org/project/tf-keras/) for more information.

## Numerical and machine-learning packages

| Package | Previous YAML | Current YAML | Relevance |
|---|---|---|---|
| TensorFlow | 2.13.0 | 2.21.0 | Neural-network training and GPU execution. |
| EnergyFlow | 1.3.2 | 1.4.0 | Constructs and trains the models used by MultiFold. |
| Keras | 2.13.1 | 3.12.4 | Major API and checkpoint behavior changes between Keras 2 and 3. |
| tf-keras | Absent | 2.21.0 | Legacy Keras implementation used in the current workflow. |
| NumPy | 1.24.3 (pip) | 1.26.4 (Conda) | Array operations; compatibility with compiled numerical packages. |
| Numba | 0.57.1 | 0.59.1 | JIT compilation; version compatibility with NumPy and llvmlite. |
| llvmlite | 0.40.1 | 0.42.0 | Compiler infrastructure used by Numba. |
| SciPy | 1.11.1 | 1.14.0 | Numerical and statistical routines. |
| h5py | 3.9.0 | 3.14.0 | HDF5 model-weight saving and loading. |
| protobuf | 4.23.4 | 7.36.2 | Serialization dependency used by TensorFlow. |
| ml-dtypes | 0.2.0 | 0.5.4 | Numerical types used by TensorFlow. |
| wrapt | 1.14.1 | 2.5.0 | Supporting dependency; check declared package compatibility. |
| JAX | 0.4.13 | Absent | Not used by the MultiFold script discussed. |

## GPU software

| Component | Previous YAML | Current YAML | Relevance |
|---|---|---|---|
| Conda cudatoolkit | 11.8.0 | 11.8.0 | Does not by itself provide the CUDA 12 libraries needed by the current TensorFlow setup. |
| TensorRT | 8.6.1.post1 | 11.4.0.106 (CUDA 13 variant) | Relevant only when used explicitly; the script's TensorRT import is commented out. |
| Pip NVIDIA CUDA packages | CUDA 11/12 packages present | CUDA 12 packages present | The updated YAML now records these GPU dependencies. |
| cuDNN | 8.6.0.163 for CUDA 11 and 8.9.4.25 for CUDA 12 | 9.27.0.42 for CUDA 12 | GPU neural-network kernels. |
| Pip CUDA runtime | 12.2.140 (CUDA 12) | 12.9.79 (CUDA 12) | CUDA runtime library loading. |

### NVIDIA packages recorded in the YAML files

| Pip package | Previous YAML | Current YAML |
|---|---|---|
| nvidia-cublas-cu11 | 11.11.3.6 | Absent |
| nvidia-cublas-cu12 | 12.2.5.6 | 12.9.2.10 |
| nvidia-cuda-cupti-cu12 | Absent | 12.9.79 |
| nvidia-cuda-nvcc-cu12 | Absent | 12.9.86 |
| nvidia-cuda-nvrtc-cu12 | 12.2.140 | 12.9.86 |
| nvidia-cuda-runtime-cu12 | 12.2.140 | 12.9.79 |
| nvidia-cudnn-cu11 | 8.6.0.163 | Absent |
| nvidia-cudnn-cu12 | 8.9.4.25 | 9.27.0.42 |
| nvidia-cufft-cu12 | Absent | 11.4.1.4 |
| nvidia-curand-cu12 | Absent | 10.3.10.19 |
| nvidia-cusolver-cu12 | Absent | 11.7.5.82 |
| nvidia-cusparse-cu12 | Absent | 12.5.10.65 |
| nvidia-nccl-cu12 | Absent | 2.32.3 |
| nvidia-nvjitlink-cu12 | Absent | 12.9.86 |

## JupyterLab packages

| Package | Previous YAML | Current YAML |
|---|---|---|
| packaging	| 23.1 | 26.3 |
| jupyterlab | Absent | 4.5.7 |
| jupyterlab_server	| Absent | 2.28.0 |
| jupyter-lsp | Absent | 2.3.1 |

## Important packages unchanged in the comparison

| Package | Previous and current YAML |
|---|---|
| Python | 3.10.8 |
| ROOT | 6.24.6 |
| pandas | 2.0.3 |
| uproot | 5.0.10 |
| awkward | 2.3.1 |
| Pythia8 | 8.307 |
| cuda-nvcc | 11.3.58 |

## Runtime and validation notes

- The current workflow uses `TF_USE_LEGACY_KERAS=1`, set before importing EnergyFlow or TensorFlow, to align the Keras model and callback implementations.
- GPU detection succeeded after adding the pip NVIDIA library directories to `LD_LIBRARY_PATH` before starting Python. Saved this setup in a separate shell setup file [setup_cuda.sh](https://github.com/y-song/multifold-101326/blob/master/setup_cuda.sh).
- A YAML export does not include custom activation/deactivation hook scripts or the submission script's library-path setup.