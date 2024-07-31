
# ContourCraft: Learning to Resolve Intersections in Neural Multi-Garment Simulations

### <img align=center src=./static/icons/project.png width='32'/> [Project](https://dolorousrtur.github.io/contourcraft/) &ensp; <img align=center src=./static/icons/paper.png width='24'/> [Paper](https://arxiv.org/abs/2405.09522) &ensp;  

This is a repository for the paper [**"ContourCraft: Learning to Resolve Intersections in Neural Multi-Garment Simulations"**](https://arxiv.org/abs/2405.09522) (SIGGRAPH2024).

It is based on and fully includes the code for the paper  [**"HOOD: Hierarchical Graphs for Generalized Modelling of Clothing Dynamics"**](https://arxiv.org/abs/2212.07242)


**NOTE:** This repo precisely follows the structure and includes all functionality of the HOOD repo. The main difference is the added code and model for **ContourCraft**. You can run inference of the **ContourCraft** model using `Inference.ipynb` or `Inference_from_any_pose.ipynb` (more details below), same as in HOOD. Soon training code for ContourCraft and more functionality specific to multi-layer outfits will be added.

TODO list:
- [x] Support for SMPL-X body model along with SMPL
- [ ] Examples of multi-layer outfits
- [ ] Automatic export to Alembic (.abc) format
- [ ] Automatic outfit re-meshing to support outfits of arbitrary resolution
- [ ] Automatic untangling procedure to combine unrelated garments into outfits

## Installation
The installation follows the procedure for HOOD, but you'll also need to install several extra libraries (see the end of this section)

### Install conda enviroment
We provide a conda environment file `hood.yml` to install all the dependencies. 
You can create and activate the environment with the following commands:

```bash
conda env create -f hood.yml
conda activate hood
```

If you want to build the environment from scratch, here are the necessary commands: 
<details>
  <summary>Build enviroment from scratch</summary>

```bash
# Create and activate a new environment
conda create -n hood python=3.9 -y
conda activate hood

# install pytorch (see https://pytorch.org/)
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y

# install pytorch_geometric (see https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
conda install pyg -c pyg -y

# install pytorch3d (see https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y
conda install pytorch3d -c pytorch3d -y


# install auxiliary packages with conda
conda install -c conda-forge munch pandas tqdm omegaconf matplotlib einops ffmpeg -y

# install more auxiliary packages with pip
pip install smplx aitviewer chumpy huepy

# create a new kernel for jupyter notebook
conda install ipykernel -y; python -m ipykernel install --user --name hood --display-name "hood"
```
</details>

### Install ContourCraft-specific libraries

#### [CCCollision](https://github.com/Dolorousrtur/CCCollisions)
This is a custom CUDA library for collision detection and response. Install using the instruction in its [README](https://github.com/Dolorousrtur/CCCollisions)

#### [NVIDIA warp](https://github.com/NVIDIA/warp)
```
pip install warp-lang
```

#### [cuGraph](https://github.com/rapidsai/cugraph) and [cudf](https://github.com/rapidsai/cudf)
Install following their instructions:
* [cuGraph](https://github.com/rapidsai/cugraph/blob/branch-24.08/docs/cugraph/source/installation/getting_cugraph.md)
* [cudf](https://github.com/rapidsai/cudf?tab=readme-ov-file#installation)


### Download data
#### HOOD data
Download the auxiliary data for HOOD using this [link](https://drive.google.com/file/d/1RdA4L6Fy50VsKZ8k7ySp5ps5YtWoHSgs/view?usp=sharing).
Unpack it anywhere you want and set the `HOOD_DATA` environmental variable to the path of the unpacked folder.
Also, set the `HOOD_PROJECT` environmental variable to the path you cloned this repository to:

```bash
export HOOD_DATA=/path/to/hood_data
export HOOD_PROJECT=/path/to/this/repository
```

#### SMPL(-X) models
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

## Inference
The jupyter notebook [Inference.ipynb](Inference.ipynb) contains an example of how to run inference of a trained HOOD model given a garment and a pose sequence.

It also has code for adding a new garment from an .obj file.

To run inference starting from arbitrary garment pose and arbitrary mesh sequence refer to the [InferenceFromMeshSequence.ipynb](Inference_from_any_pose.ipynb) notebook.  



## Repository structure
See the [RepoIntro.md](RepoIntro.md) for more details on the repository structure.



## Citation
If you use this repository in your paper, please cite:
```
      @inproceedings{grigorev2022hood,
      author = {Grigorev, Artur and Thomaszewski, Bernhard and Black, Michael J. and Hilliges, Otmar}, 
      title = {{HOOD}: Hierarchical Graphs for Generalized Modelling of Clothing Dynamics}, 
      journal = {Computer Vision and Pattern Recognition (CVPR)},
      year = {2023},
      }

      @inproceedings{grigorev2024contourcraft,
      title={{ContourCraft}: Learning to Resolve Intersections in Neural Multi-Garment Simulations},
      author={Grigorev, Artur and Becherini, Giorgio and Black, Michael and Hilliges, Otmar and Thomaszewski, Bernhard},
      booktitle={ACM SIGGRAPH 2024 Conference Papers},
      pages={1--10},
      year={2024}
      }


```