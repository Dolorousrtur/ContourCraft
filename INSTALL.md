# Installation instructions

To run ContourCraft you'll need to, first, set up the environment and, second, download required data. Here we describe both these steps.

## Set up the environment

We provide three options for the environment installation. They all should lead to exactly same environment with all the libraries needed to run ContourCraft.

### Option 1: Install from the env file:

#### Step 1: Install the environment from `ccraft.yaml`

```bash
conda env create -f  ccraft.yml
conda activate ccraft
```

#### Step 2: install additional libraries with pip:
```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" # Takes a while to build wheel
pip install loguru
```

#### Step 3: Install CCCollision

See [CCCollision](https://github.com/Dolorousrtur/CCCollisions) repository for more info.
```bash
git clone https://github.com/NVIDIA/cuda-samples.git
cd cuda-samples
export CUDA_SAMPLES_INC=$(pwd)/Common
cd ..
git clone git@github.com:Dolorousrtur/CCCollisions.git
cd CCCollisions
pip install .
```


### Option 2: Install from the spec-file

#### Step 1: Install the environment from `spec-file.txt`
```bash
conda create --name ccraft --file spec-file.txt
conda activate ccraft
```

#### Step 2: install additional libraries with pip:
```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" # Takes a while to build wheel
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
pip install torch-geometric==2.4.0
pip install warp-lang
pip install smplx aitviewer chumpy scikit-image scipy trimesh loguru
```

#### Step 3: Install CCCollision

See [CCCollision](https://github.com/Dolorousrtur/CCCollisions) repository for more info.
```bash
git clone https://github.com/NVIDIA/cuda-samples.git
cd cuda-samples
export CUDA_SAMPLES_INC=$(pwd)/Common
cd ..
git clone git@github.com:Dolorousrtur/CCCollisions.git
cd CCCollisions
pip install .
```

### Option 3: Manual environment installation
#### Step 1: Create an environment and install libraries with pip and conda

```bash
conda create -n ccraft python=3.10
conda activate ccraft
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
conda install einops -y
conda install ffmpeg -y
conda install -c conda-forge jupyterlab -y
conda install matplotlib -y
conda install munch -y
conda install networkx -y
conda install omegaconf -y
conda install pandas pillow scikit-learn tqdm yaml -y
conda install -c iopath iopath -y

pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
pip install torch-geometric==2.4.0
pip install warp-lang

conda install cudf=24.12 cugraph=24.12 -c rapidsai
pip install smplx aitviewer chumpy scikit-image scipy trimesh loguru

```

#### Step 2: Install CCCollision

See [CCCollision](https://github.com/Dolorousrtur/CCCollisions) repository for more info.
```bash
git clone https://github.com/NVIDIA/cuda-samples.git
cd cuda-samples
export CUDA_SAMPLES_INC=$(pwd)/Common
cd ..
git clone git@github.com:Dolorousrtur/CCCollisions.git
cd CCCollisions
pip install .
```


## Download data

### ContourCraft data
Download the auxiliary data for ContourCraft using this [link](https://drive.google.com/file/d/1NfxAeaC2va8TWMjiO_gbAcVPnZ8BYFPD/view?usp=sharing).
Unpack it anywhere you want and set the `DEFAULTS.data_root` variable in `defaults.py` to the path of the unpacked folder:
```python
DEFAULTS['data_root'] = '/path/to/ccraft_data'
```

Also, set the `DEFAULTS.project_dir` variable to the path you cloned this repository to:
```python
DEFAULTS['project_dir'] = '/path/to/this/repository'
```


### SMPL(-X) models

Download the SMPL-X models using this [link](https://smpl-x.is.tue.mpg.de/). Unpack them into the `DEFAULTS.data_root/aux_data/body_models/smplx` folder. For our experiments we used SMPL-X v1.1.

If you want to use SMPL models, [download them](https://smpl.is.tue.mpg.de/) and unpack into `DEFAULTS.data_root/aux_data/body_models/smpl/`. In our experiments we used SMPL of version  1.0.0 (female/male. 10 shape PCs). You'll need to rename the model files from `basicModel_*_lbs_10_207_0_v1.0.0.pkl` to `SMPL_FEMALE.pkl` and `SMPL_MALE.pkl`

In the end your `DEFAULTS.data_root` folder should look like this:
```
DEFAULTS.data_root
    |-- aux_data
        |-- datasplits // directory with csv data splits used for training the model
        |-- body_models
          |-- smpl // directory with smpl models
            |-- SMPL_FEMALE.pkl
            |-- SMPL_MALE.pkl
          |-- smplx // directory with smplx models
            |-- SMPLX_NEUTRAL.pkl
            |-- SMPLX_FEMALE.pkl
            |-- SMPLX_MALE.pkl
        |-- garment_dicts // folder with .pkl meshes for garment dictionaries use for simulation
        |-- garment_meshes // folder with .obj meshes for garments
        |-- smpl_aux.pkl // dictionary with indices of SMPL vertices that correspond to hands, used to disable hands during inference to avoid body self-intersections
    |-- trained_models // directory with trained models
        |-- hood_cvpr.pth // HOOD model used in the CVPR paper
        |-- hood_final.pth // HOOD model trained with refactored code with several bug fixes after the CVPR submission
        |-- contourcraft.pth // ContourCraft model used in the SIGGRAPH paper
    |-- examples
        |-- fromanypose  // example data to run simulation over a mesh sequence (used in the Inference_from_mesh_sequence.ipynb) 
        |-- unpose // example data for unposing germents (used in GarmentImport.ipynb)
```
## Troubleshooting

### Installing pytorch3d
Potential errors happening while running `pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"`

**Error**:

```bash

gcc: fatal error: cannot execute ‘cc1plus’: execvp: No such file or directory
```

**Solution**:

`conda install gcc=11.4.0 gxx=11.4.0`

---

**Error**:

```bash
...
site-packages/torch/include/ATen/cuda/CUDAContextLight.h:6:10: fatal error: cuda_runtime_api.h: No such file or directory
...

```

**Solution**:
```bash
# cuda_runtime_api.h whould be located in $CONDA_PREFIX/targets/x86_64-linux/include/
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include/:$CPATH
```


### Visualization with AITviewer

**Error**: 

`AttributeError: 'PyQt5Window' object has no attribute '_ctx'. Did you mean: 'ctx'`

**Solution**:

`pip install moderngl-window==2.4.6 pyglet`

---
**Error**: 

`OSError: libGL.so: cannot open shared object file: No such file or directory`

**Solution**:
Create a symbolic link from `libGL.so` to `libGL.so.1`
```bash
locate libGL.so # should give you the path to libGL.so.1
sudo ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so
```