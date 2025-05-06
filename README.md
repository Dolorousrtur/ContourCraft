
# ContourCraft: Learning to Resolve Intersections in Neural Multi-Garment Simulations

### <img align=center src=./static/icons/project.png width='32'/> [Project](https://dolorousrtur.github.io/contourcraft/) &ensp; <img align=center src=./static/icons/paper.png width='24'/> [Paper](https://arxiv.org/abs/2405.09522) &ensp;  

This is a repository for the paper [**"ContourCraft: Learning to Resolve Intersections in Neural Multi-Garment Simulations"**](https://arxiv.org/abs/2405.09522) (SIGGRAPH2024).


**NOTE:** This repo precisely follows the structure and includes all functionality of the [HOOD repo](https://github.com/dolorousrtur/hood). The main difference is the added code and model for **ContourCraft**. You can run inference of the **ContourCraft** model using `Inference.ipynb` or `Inference_from_mesh_sequence.ipynb` (more details below), same as in HOOD. 

TODO list:
- [x] Support for SMPL-X body model along with SMPL
- [x] Examples of multi-layer outfits
- [x] Automatic untangling procedure to combine unrelated garments into outfits
- [ ] Automatic export to Alembic (.abc) format
- [ ] Automatic outfit re-meshing to support outfits of arbitrary resolution

## Installation
Follow [INSTALL.md](INSTALL.md) to install the environment and download data required for running ContourCraft.

## Inference
The jupyter notebook [Inference.ipynb](Inference.ipynb) contains an example of how to run inference of a trained ContourCraft model given a garment and a pose sequence.

To run inference starting from arbitrary garment pose and arbitrary mesh sequence refer to the [Inference_from_mesh_sequence.ipynb](Inference_from_mesh_sequence.ipynb) notebook.  

To convert new garments to the format used in ContourCraft, refer to [GarmentImport.ipynb](GarmentImport.ipynb)



## Training
Follow the instructions in [TRAINING.md](TRAINING.md) to train a ContourCraft model from scratch.

## Gaussian Garments
To finetune a ContourCraft model with GaussianGarments registrations and then simulate garment sequences with the finetuned behavior, refer to [GaussianGarments.ipynb](GaussianGarments.ipynb) 

## Repository structure
The repository structure closely follows the repository for HOOD. Please see [RepoIntro.md](https://github.com/Dolorousrtur/HOOD/blob/main/RepoIntro.md)  in the HOOD repository for more details.




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
