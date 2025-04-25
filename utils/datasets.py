from pathlib import Path
import pickle
from typing import Dict

import torch
import smplx
from smplx import SMPL
import os
from utils.defaults import DEFAULTS

from utils.garment_smpl import GarmentSMPL


def convert_lbs_dict(lbs_dict: dict) -> Dict[str, torch.FloatTensor]:
    """ Convert the values of the lbs_dict from np.array to torch.Tensor"""

    for k in ['shapedirs', 'posedirs', 'lbs_weights', 'v']:
        lbs_dict[k] = torch.FloatTensor(lbs_dict[k])

    return lbs_dict


def load_garments_dict(path: str) -> Dict[str, dict]:
    """ Load the garments_dict containing data for all garments from a pickle file"""

    with open(path, 'rb') as f:
        garments_dict = pickle.load(f)

    for garment, g_dict in garments_dict.items():
        g_dict['lbs'] = convert_lbs_dict(g_dict['lbs'])

    return garments_dict


def make_garment_smpl_dict(garments_dict: Dict[str, dict], smpl_model: SMPL) -> Dict[str, GarmentSMPL]:
    """ For each garment create a GarmentSMPL object"""
    garment_smpl_model_dict = dict()
    for garment, g_dict in garments_dict.items():
        g_smpl_model = GarmentSMPL(smpl_model, g_dict['lbs'])
        garment_smpl_model_dict[garment] = g_smpl_model

    return garment_smpl_model_dict


def make_fromanypose_dataloader(pose_sequence_type, pose_sequence_path, garment_template_path, smpl_model=None):
    from datasets.from_any_pose import Config as DatasetConfig
    from datasets.from_any_pose import create as create_dataset
    from utils.dataloader import DataloaderModule
    from utils.arguments import DataConfig as DataloaderConfig

    config = DatasetConfig(pose_sequence_type=pose_sequence_type, 
                        pose_sequence_path=pose_sequence_path, 
                        garment_template_path=garment_template_path)


    dataset = create_dataset(config)
    dataloader_config = DataloaderConfig(num_workers=0)
    dataloader = DataloaderModule(dataset, dataloader_config).create_dataloader()
    return dataloader

def build_smpl_bygender(smpl_root, model_type='smpl', use_pca=False):
    smpl_root = Path(smpl_root)
    smpl_model_dict = {}

    for gender in ['male', 'female', 'neutral']:
        smpl_model_dict[gender] = smplx.create(smpl_root, model_type=model_type, gender=gender, use_pca=use_pca)
    
    return smpl_model_dict


def make_obstacle_dict(mcfg) -> dict:
    if mcfg.obstacle_dict_file is None:
        return {}

    obstacle_dict_path = os.path.join(DEFAULTS.aux_data, mcfg.obstacle_dict_file)
    with open(obstacle_dict_path, 'rb') as f:
        obstacle_dict = pickle.load(f)
    return obstacle_dict
