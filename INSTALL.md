# Installation instructions

To run contourCraft you'll need to, first, set up the environment and, second, download required data. Here we descrime both these steps.

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

### HOOD data
Download the auxiliary data for HOOD using this [link](https://drive.google.com/file/d/1RdA4L6Fy50VsKZ8k7ySp5ps5YtWoHSgs/view?usp=sharing).
Unpack it anywhere you want and set the `HOOD_DATA` environmental variable to the path of the unpacked folder.
Also, set the `HOOD_PROJECT` environmental variable to the path you cloned this repository to:

```bash
export HOOD_DATA=/path/to/hood_data
export HOOD_PROJECT=/path/to/this/repository
```

### SMPL(-X) models
Download the SMPL models using this [link](https://smpl.is.tue.mpg.de/). Unpack them into the `$HOOD_DATA/aux_data/body_models/smpl` folder.

If you want to use SMPL-X models, [download them](https://smpl-x.is.tue.mpg.de/) and unpack into `$HOOD_DATA/aux_data/body_models/smplx`.

In the end your `$HOOD_DATA` folder should look like this:
```
$HOOD_DATA
    |-- aux_data
        |-- datasplits // directory with csv data splits used for training the model
        |-- body_models
          |-- smpl // directory with smpl models
            |-- SMPL_NEUTRAL.pkl
            |-- SMPL_FEMALE.pkl
            |-- SMPL_MALE.pkl
          |-- smplx // directory with smplx models
            |-- SMPLX_NEUTRAL.pkl
            |-- SMPLX_FEMALE.pkl
            |-- SMPLX_MALE.pkl
        |-- garment_meshes // folder with .obj meshes for garments used in HOOD
        |-- garments_dict.pkl // dictionary with garmentmeshes and their auxilliary data used for training and inference
        |-- smpl_aux.pkl // dictionary with indices of SMPL vertices that correspond to hands, used to disable hands during inference to avoid body self-intersections
    |-- trained_models // directory with trained HOOD models
        |-- cvpr_submission.pth // model used in the CVPR paper
        |-- postcvpr.pth // model trained with refactored code with several bug fixes after the CVPR submission
        |-- fine15.pth // baseline model without denoted as "Fine15" in the paper (15 message-passing steps, no long-range edges)
        |-- fine48.pth // baseline model without denoted as "Fine48" in the paper (48 message-passing steps, no long-range edges)
```
## Troubleshooting

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