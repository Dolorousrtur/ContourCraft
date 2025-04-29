import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from omegaconf import OmegaConf, MISSING, DictConfig
from torch.utils.data import DataLoader

from utils.arguments import create_module, create_runner, create_dataloader_modules, load_module
from utils.defaults import DEFAULTS

from utils.dataloader import DataloaderModule
from utils.arguments import DataConfig as DataloaderConfig



def update_config_single_sequence(experiment_config, sequence_path, garment_name):
    """
    Update the experiment config loaded from .yaml file to use it for single sequence inference.
    :param experiment_config: OmegaConf config loaded from .yaml file
    :param sequence_path: path to the SMPL pose sequence file
    :param garment_name: name of the garment to be used for inference
    :return: updated experiment config
    """
    dataset_name = list(experiment_config.dataloader.dataset.keys())[0]

    data_root, file_name = os.path.split(sequence_path)
    file_name, ext = os.path.splitext(file_name)

    experiment_config.dataloader.dataset[dataset_name].data_root = data_root
    experiment_config.dataloader.dataset[dataset_name].single_sequence_file = file_name
    experiment_config.dataloader.dataset[dataset_name].single_sequence_garment = garment_name

    return experiment_config

def apply_material_params(experiment_config, material_dict):
    runner_name = list(experiment_config.runner.keys())[0]
    experiment_config.runner[runner_name].material.density_override = material_dict['density']
    experiment_config.runner[runner_name].material.lame_mu_override = material_dict['lame_mu']
    experiment_config.runner[runner_name].material.lame_lambda_override = material_dict['lame_lambda']
    experiment_config.runner[runner_name].material.bending_coeff_override = material_dict['bending_coeff']
    return experiment_config


def _load_runner_from_state_dict(modules, experiment_config, state_dict):
    runner_module, runner, _ = create_runner(modules, experiment_config,
                                             create_aux_modules=False)
    state_dict_runner = state_dict['training_module']
    runner.load_state_dict(state_dict_runner)
    return runner_module, runner

def load_runner_from_checkpoint(checkpoint_path: str, modules: dict, experiment_config: DictConfig):
    """
    Builds a Runner objcect
    :param checkpoint_path: path to the checkpoint to load
    :param modules: dictionary  of .py modules (from utils.arguments.load_params())
    :param experiment_config: OmegaConf config for the experiment
    :return: runner_module: .py module containing the Runner class
                runner: Runner object
    """
    sd = torch.load(checkpoint_path)
    runner_module, runner = _load_runner_from_state_dict(modules, experiment_config, sd)

    return runner_module, runner


def _load_material_from_state_dict(experiment_config, state_dict):
    material_stack_m = load_module('material', experiment_config.material_stack)
    material_stack = create_module(material_stack_m, experiment_config.material_stack)
    material_stack.load_state_dict(state_dict['material_stack'])
    return material_stack

def load_runner_and_material_from_checkpoint(checkpoint_path: str, modules: dict, experiment_config: DictConfig):

    sd = torch.load(checkpoint_path)
    runner_module, runner = _load_runner_from_state_dict(modules, experiment_config, sd)
    material_stack = _load_material_from_state_dict(experiment_config, sd)
    return runner_module, runner, material_stack


def create_one_sequence_dataloader(use_config=None, dataset_name=None, dataloader_type='inference', **kwargs) -> DataLoader:

    if use_config is not None:
        config_dir = Path(DEFAULTS.project_dir) / 'configs'
        config_path = os.path.join(config_dir, use_config + '.yaml')
        conf_file = OmegaConf.load(config_path)


        dataset_name = list(conf_file.dataloaders[dataloader_type].dataset.keys())[0]
        dataset_config_dict = conf_file.dataloaders[dataloader_type].dataset[dataset_name]
    else:
        dataset_config_dict = {}

    dataset_module = importlib.import_module(f'datasets.{dataset_name}')

    DatasetConfig = dataset_module.Config
    create_dataset = dataset_module.create

    dataset_config_dict.update(kwargs)
    config = DatasetConfig(**dataset_config_dict)

    dataset = create_dataset(config)
    dataloader_config = DataloaderConfig(num_workers=0)
    dataloader = DataloaderModule(dataset, dataloader_config).create_dataloader()
    return dataloader

def create_postcvpr_one_sequence_dataloader(sequence_path: str, garment_name: str, config=None, **kwargs) -> DataLoader:

    data_root, file_name = os.path.split(sequence_path)
    file_name, _ = os.path.splitext(file_name)

    if config is None:
        config = 'contourcraft'


    dataloader = create_one_sequence_dataloader(use_config=config, data_root=data_root, single_sequence_file=file_name,
                                            single_sequence_garment=garment_name,
                                            **kwargs)
    
    return dataloader




def create_fromanypose_dataloader(pose_sequence_type, pose_sequence_path, garment_template_path, **kwargs) -> DataLoader:
    dataloader = create_one_sequence_dataloader(dataset_name='from_any_pose', pose_sequence_type=pose_sequence_type,
                                                pose_sequence_path=pose_sequence_path, garment_template_path=garment_template_path,
                                                **kwargs)
    
    return dataloader
    


def make_fromanypose_dataloader(pose_sequence_type, pose_sequence_path, garment_template_path, smpl_model=None):
    from datasets.from_any_pose import Config as DatasetConfig
    from datasets.from_any_pose import create as create_dataset
    from utils.dataloader import DataloaderModule
    from utils.arguments import DataConfig as DataloaderConfig

    config = DatasetConfig(pose_sequence_type=pose_sequence_type, 
                        pose_sequence_path=pose_sequence_path, 
                        garment_template_path=garment_template_path, 
                        smpl_model=smpl_model)


    dataset = create_dataset(config)
    dataloader_config = DataloaderConfig(num_workers=0)
    dataloader = DataloaderModule(dataset, dataloader_config).create_dataloader()
    return dataloader



def replace_model(modules: dict, current_config: DictConfig, model_config_name: str, config_dir: str=None):


    if config_dir is None:
        config_dir = Path(DEFAULTS.project_dir) / 'configs'

    model_config_path = os.path.join(config_dir, model_config_name + '.yaml')
    model_config = OmegaConf.load(model_config_path)

    current_config.model = OmegaConf.merge(model_config.model, current_config.model)
    modules['model'] = load_module('models', current_config.model)

    return modules, current_config
