# vtopcd3 == vto2_barydeci_pcd_multi
from collections import defaultdict
import importlib
import os
import pickle
from dataclasses import dataclass, MISSING
from functools import partial
from pathlib import Path
from random import random
from typing import Optional, Dict

import numpy as np
import pandas as pd
import smplx
import torch
import torch_geometric
from smplx import SMPL
from torch_geometric.data import HeteroData

from utils.cloth_and_material import load_obj
from utils.coarse import make_coarse_edges
from utils.common import NodeType, pickle_dump, pickle_load, triangles_to_edges, separate_arms
from utils.datasets import make_obstacle_dict, build_smpl_bygender
from utils.defaults import DEFAULTS
from scipy.spatial.transform import Rotation as R
from datasets.ccraft import VertexBuilder as VertexBuilderCCraft
from datasets.ccraft import GarmentBuilder as GarmentBuilderBase 
from datasets.ccraft import BodyBuilder as BodyBuilderBase
from datasets.ccraft import NoiseMaker, create_loader
import json


@dataclass
class Config:
    train_split_path: str = MISSING
    valid_split_path: str = MISSING
    smpl_dir: str = MISSING
    garment_dict_dir: str = MISSING
    smplx_segmentation_file: Optional[str] = None

    registration_root: Optional[str] = None # do not set
    data_root: Optional[str] = None # do not set
    temp_data_root: Optional[str] = None # do not set

    body_model_root: str = 'body_models'  # Path to the directory containg body model files, should contain `smpl` and/or `smplx` sub-directories. Relative to $DEFAULTS.data_root/aux_data/
    model_type: str = 'smpl'  # Type of the body model ('smpl' or 'smplx')
    sequence_loader: str = 'hood_pkl'  # Name of the sequence loader to use 

    pinned_verts: bool = True
    use_betas_for_restpos: bool = False
    n_coarse_levels: int = 4
    separate_arms: bool = True
    button_edges: bool = True
    omit_hands: bool = True
    nobody_freq: float = 0.
    
    wholeseq: bool = True

    noise_scale: float = 0.0
    restpos_scale_max: float = 1.0
    restpos_scale_min: float = 1.0
    use_betas_for_restpos: bool = False




def create(mcfg: Config, **kwargs):

    mcfg.registration_root = DEFAULTS.gaga.registration_root
    mcfg.data_root = DEFAULTS.gaga.data_root
    mcfg.temp_data_root = DEFAULTS.gaga.temp_data_root

    loader = create_loader(mcfg)

    if 'valid' in kwargs and kwargs['valid']:
        split_file = mcfg.valid_split_path
    else:
        split_file = mcfg.train_split_path

    split_path = os.path.join(DEFAULTS.aux_root, split_file)
    datasplit = pd.read_csv(split_path, dtype='str')

    dataset = Dataset(loader, datasplit)
    return dataset



class VertexBuilder(VertexBuilderCCraft):
    def __init__(self, mcfg):
        self.mcfg = mcfg

    @staticmethod
    def build(sequence_dict, f_make, idx_start, idx_end=None, garment_name=None):
        betas = sequence_dict['betas']
        if len(betas.shape) == 2 and betas.shape[0] != 1:
            betas = betas[idx_start: idx_end]


        ks = ['body_pose', 'global_orient', 'transl', 'expression', 'jaw_pose', 'leye_pose', 'reye_pose', 'left_hand_pose', 'right_hand_pose']
        
        sequence_dict_frames = {k: sequence_dict[k][idx_start: idx_end] for k in ks}
        sequence_dict_frames['betas'] = betas
        sequence_dict_frames['type'] = sequence_dict['type']

        verts = f_make(sequence_dict_frames, garment_name=garment_name)

        return verts


    


class GarmentBuilder:
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

    
    def load_mesh_sequence(self, sequence_dir):
        sequence_dir = Path(sequence_dir)

        reg_gname = sequence_dir.parts[-3]
        take = sequence_dir.parts[-2]

        temp_path = Path(self.mcfg.temp_data_root) / 'garment'/ reg_gname / (take + '.pkl')
        temp_path.parent.mkdir(parents=True, exist_ok=True)

        if temp_path.exists():
            verts, faces = pickle_load(temp_path)
            return verts, faces


        from_frames = list(sequence_dir.glob('*.obj'))
        from_frames = sorted(from_frames, key=lambda x: int(x.stem.split('_')[-1]))

        faces = None
        verts = []

        for frame in from_frames:
            v, f = load_obj(frame)


            verts.append(v)
            if faces is None:
                faces = f

            if len(verts) > 0 and v.shape != verts[0].shape:
                print('normal shape:', verts[0].shape)
                print(frame, v.shape)

        verts = np.stack(verts, axis=0)


        pickle_dump((verts, faces), temp_path)

        return verts, faces      
    
    def add_verts(self, sample, verts):
        all_verts = torch.tensor(verts).permute(1, 0, 2)

        prev_pos = all_verts[:, :-2]
        pos = all_verts[:, 1:-1]
        target_pos = all_verts[:, 2:]

        sample['cloth'].pos = pos
        sample['cloth'].prev_pos = prev_pos
        sample['cloth'].target_pos = target_pos

        return sample

    def build(self, sample, sequence_dir, garment_name, sequence_dict):
        sample_temp = HeteroData()

        verts, faces = self.load_mesh_sequence(sequence_dir)
        sample_temp = self.add_verts(sample_temp, verts)


        sample_temp = self.add_vertex_type(sample_temp, garment_name)
        sample_temp = self.add_restpos(sample_temp, sequence_dict, garment_name)
        sample_temp = self.add_faces_and_edges(sample_temp, garment_name)
        sample_temp = self.add_coarse(sample_temp, garment_name)
        sample_temp = self.add_garment_id(sample_temp, garment_name)
        sample_temp = self.scale_restpos(sample_temp, garment_name)

        sample = self.add_garment_to_sample(sample, sample_temp)

        return sample


    
class PKLFolderSequenceLoader():

    def __init__(self, mcfg):
        self.mcfg = mcfg

    def collect_poseseq(self, sequence_dir):

        from_frames = list(sequence_dir.glob('*.pkl'))
        from_frames = sorted(from_frames, key=lambda x: int(x.stem.split('_')[-1]))

        pose_dict = defaultdict(list)
        for frame in from_frames:
            pose = pickle_load(frame)

            for k, v in pose.items():
                pose_dict[k].append(v)

        for k, v in pose_dict.items():
            pose_dict[k] = np.stack(v, axis=0)       

        return pose_dict

    def load_sequence(self, sequence_dir: str):
        sequence_dir = Path(sequence_dir)

        subject = sequence_dir.parts[-5]
        garment_name = sequence_dir.parts[-4]
        take = sequence_dir.parts[-3]

        temp_path = Path(self.mcfg.temp_data_root) / 'body'/ subject / garment_name / (take + '.pkl')
        temp_path.parent.mkdir(parents=True, exist_ok=True)


        if temp_path.exists():
            sequence = pickle_load(temp_path)
            return sequence
            
        sequence = self.collect_poseseq(sequence_dir)
        pickle_dump(sequence, temp_path)
        return sequence

class Loader:
    def __init__(self, mcfg: Config, body_models_dict: dict, garment_dicts_dir: str, obstacle_dict: dict, betas_table=None):
        
        sequence_loader_module = importlib.import_module(f'datasets.sequence_loaders.{mcfg.sequence_loader}')
        SequenceLoader = sequence_loader_module.SequenceLoader

        self.sequence_loader = SequenceLoader(mcfg, mcfg.data_root, betas_table=betas_table)
        self.garment_builder = GarmentBuilder(mcfg, body_models_dict, garment_dicts_dir)
        self.body_builder = BodyBuilderBase(mcfg, body_models_dict, obstacle_dict)

        self.data_path = mcfg.data_root

        self.mcfg = mcfg

    def load_sample(self, subject, garment_name, reg_gname, take, gender):

        smplx_sequence_path = os.path.join(self.seq_root, subject, garment_name, take, 'Meshes', 'smplx')
        smpl_sequence = self.sequence_loader.load_sequence(smplx_sequence_path)

        sample = HeteroData()
        sample = self.body_builder.build(sample, smpl_sequence, gender)


        garment_seq_dir = os.path.join(self.registration_root, reg_gname, take.lower(), 'meshes')
        sample = self.garment_builder.build(sample, garment_seq_dir, reg_gname, smpl_sequence)



        return sample

class Dataset:
    def __init__(self, loader: Loader, datasplit: pd.DataFrame, wholeseq=False):
        """
        Dataset class for building training and validation samples
        :param loader: Loader object
        :param datasplit: pandas DataFrame with the following columns:
            id: sequence name relative to loader.data_path
            length: number of frames in the sequence
            garment: name of the garment
        :param wholeseq: if True, load the whole sequence, otherwise load a single frame
        """

        self.loader = loader
        self.datasplit = datasplit
        self.wholeseq = wholeseq

        if self.wholeseq:
            self._len = self.datasplit.shape[0]
        else:
            all_lens = datasplit.length.tolist()
            self.all_lens = [int(x) - 7 for x in all_lens]
            self._len = sum(self.all_lens)

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

        betas_id = None
        if self.wholeseq:
            fname = self.datasplit.id[item]
            garment_name = self.datasplit.garment[item]
            if 'gender' in self.datasplit:
                gender = self.datasplit.gender[item]
            else:
                gender = 'female'
            idx = 0

            if 'betas_id' in self.datasplit:
                betas_id = int(self.datasplit.betas_id[item])
        else:
            fname, idx, garment_name, gender = self._find_idx(item)

        sample = self.loader.load_sample(fname, idx, garment_name, gender, betas_id=betas_id)
        sample['sequence_name'] = fname
        sample['garment_name'] = garment_name

        return sample

    def __len__(self) -> int:
        return self._len
