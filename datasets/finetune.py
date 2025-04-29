import importlib
import os
from dataclasses import dataclass, MISSING
from pathlib import Path
from typing import Optional, Dict, Tuple

import pandas as pd
import torch
from torch_geometric.data import HeteroData

from utils.datasets import build_smpl_bygender, make_obstacle_dict
from utils.io import pickle_load
from utils.defaults import DEFAULTS
from datasets.ccraft import GarmentBuilder as GarmentBuilderBase 
from datasets.ccraft import BodyBuilder as BodyBuilder


@dataclass
class Config:
    train_split_path: str = MISSING
    valid_split_path: str = MISSING
    smpl_dir: str = MISSING
    garment_dict_dir: str = MISSING
    registration_root: Optional[str] = MISSING
    body_sequence_root: Optional[str] = MISSING
    smplx_segmentation_file: Optional[str] = None

    # data_root: Optional[str] = None # do not set
    # temp_data_root: Optional[str] = None # do not set
    obstacle_dict_file: Optional[str] = None

    body_model_root: str = 'body_models'  # Path to the directory containg body model files, should contain `smpl` and/or `smplx` sub-directories. Relative to $DEFAULTS.data_root/aux_data/
    model_type: str = 'smpl'  # Type of the body model ('smpl' or 'smplx')
    sequence_loader: str = 'hood_pkl'  # Name of the sequence loader to use 

    swap_axes: bool = False
    pinned_verts: bool = True
    use_betas_for_restpos: bool = False
    n_coarse_levels: int = 4
    separate_arms: bool = True
    button_edges: bool = True
    omit_hands: bool = True
    nobody_freq: float = 0.
    fps: int = 30
    wholeseq: bool = True

    repeat_datasplit: int = 100
    
    noise_scale: float = 0.0
    restpos_scale_max: float = 1.0
    restpos_scale_min: float = 1.0
    use_betas_for_restpos: bool = False

    single_sequence_file: Optional[str] = None

def create_loader(mcfg: Config):
    garment_dict_dir = Path(DEFAULTS.aux_data) / mcfg.garment_dicts_dir
    body_model_root = Path(DEFAULTS.aux_data) / mcfg.body_model_root

    if mcfg.sequence_loader == 'hood_pkl':
        mcfg.model_type = 'smpl'
    elif 'smpl' in mcfg.sequence_loader == 'cmu_npz_':
        mcfg.model_type = 'smpl'
    elif 'smplx' in  mcfg.sequence_loader:
        mcfg.model_type = 'smplx'

    body_models_dict = build_smpl_bygender(body_model_root, mcfg.model_type)
    obstacle_dict = make_obstacle_dict(mcfg)

    loader = Loader(mcfg, 
                    body_models_dict, garment_dict_dir, obstacle_dict=obstacle_dict)
    return loader



def create(mcfg: Config, **kwargs):
    loader = create_loader(mcfg)

    if 'valid' in kwargs and kwargs['valid']:
        split_file = mcfg.valid_split_path
    else:
        split_file = mcfg.train_split_path

    split_path = os.path.join(DEFAULTS.aux_data, split_file)
    datasplit = pd.read_csv(split_path, dtype='str')

    if 'valid' not in kwargs or not kwargs['valid']:
        datasplit = pd.concat([datasplit] * mcfg.repeat_datasplit, ignore_index=True)

    dataset = Dataset(loader, datasplit)
    return dataset

    


class GarmentBuilder(GarmentBuilderBase):
    def __init__(self, mcfg: Config, body_models_dict, garment_dicts_dir: str):
        super().__init__(mcfg, body_models_dict, garment_dicts_dir)
        


    
    def scale_restpos(self, sample, garment_name):
        garment_dict = self.garments_dict[garment_name]
        if 'rest_pos_scale' not in garment_dict:
            return sample
        
        rest_pos_scale = torch.FloatTensor(garment_dict['rest_pos_scale']).unsqueeze(1)
        rest_pos = sample['cloth'].rest_pos

        rest_pos = rest_pos * rest_pos_scale
        sample['cloth'].rest_pos = rest_pos

        return sample

    
    def add_uvs(self, sample, garment_name: str):
        garment_dict = self.garments_dict[garment_name]
        if 'uv_coords' not in garment_dict:
            return sample
        
        uv_coords = garment_dict['uv_coords']
        sample['cloth'].uv_coords = torch.FloatTensor(uv_coords)


        uv_faces = garment_dict['uv_faces']
        uv_faces = torch.LongTensor(uv_faces)
        sample['cloth'].uv_faces_batch = torch.LongTensor(uv_faces).T


        return sample

    
    def load_mesh_sequence(self, sequence_path):
        mesh_sequence = pickle_load(sequence_path)
        verts = mesh_sequence['vertices']
        faces = mesh_sequence['faces']
        return verts, faces      
    
    def add_verts(self, sample, verts):
        all_verts = torch.tensor(verts).permute(1, 0, 2)

        sample['cloth'].pos = all_verts[:, 0]
        sample['cloth'].prev_pos = all_verts[:, 1]
        sample['cloth'].target_pos = all_verts[:, 2]
        sample['cloth'].lookup = all_verts[:, 2:]

        return sample

    def build(self, sample, sequence_path, garment_name, sequence_dict):
        self.load_garment_dict(garment_name)
        sample_temp = HeteroData()

        verts, faces = self.load_mesh_sequence(sequence_path)
        sample_temp = self.add_verts(sample_temp, verts)


        sample_temp = self.add_vertex_type(sample_temp, garment_name)
        sample_temp = self.add_restpos(sample_temp, sequence_dict, garment_name)
        sample_temp = self.add_faces_and_edges(sample_temp, garment_name)
        sample_temp = self.add_coarse(sample_temp, garment_name)
        sample_temp = self.add_garment_id(sample_temp, garment_name)
        sample_temp = self.scale_restpos(sample_temp, garment_name)

        sample = self.add_garment_to_sample(sample, sample_temp)

        return sample


class Loader:
    def __init__(self, mcfg: Config, body_models_dict: dict, garment_dicts_dir: str, obstacle_dict: dict, betas_table=None):
        
        sequence_loader_module = importlib.import_module(f'datasets.sequence_loaders.{mcfg.sequence_loader}')
        SequenceLoader = sequence_loader_module.SequenceLoader

        self.sequence_loader = SequenceLoader(mcfg, mcfg.body_sequence_root, betas_table=betas_table)
        self.garment_builder = GarmentBuilder(mcfg, body_models_dict, garment_dicts_dir)
        self.body_builder = BodyBuilder(mcfg, body_models_dict, obstacle_dict)

        self.mcfg = mcfg

    def load_sample(self, subject, sequence, gender):

        smplx_sequence_path = Path(self.mcfg.body_sequence_root) / (sequence + '.npz')
        smpl_sequence = self.sequence_loader.load_sequence(smplx_sequence_path)

        sample = HeteroData()
        sample = self.body_builder.build(sample, smpl_sequence, 0, gender)

        garment_seq_path = Path(self.mcfg.registration_root) / (sequence + '.pkl')
        sample = self.garment_builder.build(sample, garment_seq_path, subject, smpl_sequence)

        return sample

class Dataset:
    def __init__(self, loader: Loader, datasplit: pd.DataFrame):
        """
        Dataset class for building training and validation samples
        :param loader: Loader object
        :param datasplit: pandas DataFrame with the following columns:
            id: sequence name relative to loader.data_path
            length: number of frames in the sequence
            garment: name of the garment
        """

        self.loader = loader
        self.datasplit = datasplit
        self._len = self.datasplit.shape[0] 

    def _find_idx(self, index: int) -> Tuple[str, int, str]:
        """
        Takes a global index and returns the sequence name, frame index and garment name for it
        """
        fi = 0
        while self.all_lens[fi] <= index:
            index -= self.all_lens[fi]
            fi += 1
        if 'gender' in self.datasplit:
            gender = self.datasplit.gender[fi]
        else:
            gender = 'female'

        return self.datasplit.id[fi], index, self.datasplit.garment[fi], gender


    def __getitem__(self, item: int) -> HeteroData:
        """
        Load a sample given a global index
        """

        fname = self.datasplit.id[item]
        garment_name = self.datasplit.garment[item]
        if 'gender' in self.datasplit:
            gender = self.datasplit.gender[item]
        else:
            gender = 'female'
        idx = 0


        sample = self.loader.load_sample(garment_name, fname, gender)
        sample['sequence_name'] = fname
        sample['garment_name'] = garment_name

        return sample

    def __len__(self) -> int:
        return self._len
