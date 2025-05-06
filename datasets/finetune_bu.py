# vtopcd3 == vto2_barydeci_pcd_multi
from collections import defaultdict
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
    garment_dict_file: str = MISSING
    smplx_segmentation_file: Optional[str] = None


    registration_root: Optional[str] = None # do not set
    data_root: Optional[str] = None # do not set
    temp_data_root: Optional[str] = None # do not set

    pinned_verts: bool = True
    use_betas_for_restpos: bool = False
    n_coarse_levels: int = 4
    separate_arms: bool = True
    button_edges: bool = True
    omit_hands: bool = True
    nobody_freq: float = 0.

    omit_garments: Optional[str] = None
    keep_only: Optional[str] = None



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


    def pos2tensor(self, pos):
        pos = torch.tensor(pos).permute(1, 0, 2)
        return pos

    def make_seqdict_noglort(self, sequence_dict):
        sequence_dict_noglort = {k: v for k, v in sequence_dict.items()}
        sequence_dict_noglort['global_orient'] = sequence_dict_noglort['global_orient'] * 0
        sequence_dict_noglort['transl'] = sequence_dict_noglort['transl'] * 0
        return sequence_dict_noglort

    def add_verts(self, sample, sequence_dict, f_make, object_key, **kwargs):


        N_steps = sequence_dict['body_pose'].shape[0]
        pos_dict = {}
        aux_dict = {}

        idx_from = 0
        idx_to = N_steps

        all_vertices = VertexBuilder.build(sequence_dict, f_make, idx_from, idx_to,
                                           **kwargs)
        

        pos_dict['prev_pos'] = all_vertices[:-2]
        pos_dict['pos'] = all_vertices[1:-1]
        pos_dict['target_pos'] = all_vertices[2:]


        for k, v in pos_dict.items():
            v = self.pos2tensor(v)
            setattr(sample[object_key], k, v)

        return sample, aux_dict


class GarmentBuilder:
    def __init__(self, mcfg, garments_dict):
        self.mcfg = mcfg
        self.garments_dict = garments_dict
        self.vertex_builder = VertexBuilder(mcfg)
        self.noise_maker = NoiseMaker(mcfg)

    def add_vertex_type(self, sample, garment_name: str):
        """
        Add `vertex_type` tensor to `sample['cloth']`

        utils.common.NodeType.NORMAL (0) for normal vertices
        utils.common.NodeType.HANDLE (3) for pinned vertices

        if `self.mcfg.pinned_verts` == True, take `vertex_type` from `self.garments_dict`
        else: fill all with utils.common.NodeType.NORMAL (0)

        :param sample: HeteroData sample
        :param garment_name: name of the garment in `self.garments_dict`

        :return: sample['cloth'].vertex_type: torch.LongTensor [Vx1]
        """
        garment_dict = self.garments_dict[garment_name]
        V = sample['cloth'].pos.shape[0]

        if self.mcfg.pinned_verts:
            vertex_type = garment_dict['node_type'].astype(np.int64)
        else:
            vertex_type = np.zeros((V, 1)).astype(np.int64)

        sample['cloth'].vertex_type = torch.tensor(vertex_type)
        return sample


    def make_shaped_restpos(self, sequence_dict: dict, garment_name: str) -> np.array:
        """
        Create resting pose geometry for a garment in SMPL zero pose with given SMPL betas

        :param sequence_dict: dict with
            sequence_dict['body_pose'] np.array SMPL body pose [Nx69]
            sequence_dict['global_orient'] np.array SMPL global_orient [Nx3]
            sequence_dict['transl'] np.array SMPL translation [Nx3]
            sequence_dict['betas'] np.array SMPL betas [10]
        :param garment_name: name of the garment in `self.garment_smpl_model_dict`
        :return: zeroposed garment with given shape [Vx3]

        """


        smplx_dict_zeropos = {}
        smplx_dict_zeropos['body_pose'] = np.zeros_like(sequence_dict['body_pose'][:1])
        smplx_dict_zeropos['global_orient'] = np.zeros_like(sequence_dict['global_orient'][:1])
        smplx_dict_zeropos['transl'] = np.zeros_like(sequence_dict['transl'][:1])
        smplx_dict_zeropos['betas'] = sequence_dict['betas'][:1]
        smplx_dict_zeropos['type'] = sequence_dict['type']


        verts = self.make_cloth_verts(smplx_dict_zeropos, garment_name=garment_name)
        return verts

    def add_restpos(self, sample, sequence_dict: dict, garment_name: str):
        """
        Add resting pose geometry to `sample['cloth']`

        :param sample: HeteroData
        :param sequence_dict: dict with SMPL parameters
        :param garment_name: name of the garment in `self.garment_smpl_model_dict` and `self.garments_dict`
        :return: sample['cloth'].rest_pos: torch.FloatTensor [Vx3]
        """
        garment_dict = self.garments_dict[garment_name]

        if self.mcfg.use_betas_for_restpos:
            rest_pos = self.make_shaped_restpos(sequence_dict, garment_name)[0]
        else:
            rest_pos = garment_dict['rest_pos']

        sample['cloth'].rest_pos = torch.tensor(rest_pos)
        return sample
    
    def scale_restpos(self, sample, garment_name):
        garment_dict = self.garments_dict[garment_name]
        if 'rest_pos_scale' not in garment_dict:
            return sample
        
        rest_pos_scale = torch.FloatTensor(garment_dict['rest_pos_scale']).unsqueeze(1)
        rest_pos = sample['cloth'].rest_pos

        rest_pos = rest_pos * rest_pos_scale
        sample['cloth'].rest_pos = rest_pos

        return sample



    def add_faces_and_edges(self, sample, garment_name):
        """
        Add garment faces to `sample['cloth']`
        Add bi-directional edges to `sample['cloth', 'mesh_edge', 'cloth']`

        :param sample: HeteroData
        :param garment_name: name of the garment in `self.garment_smpl_model_dict` and `self.garments_dict`
        :return:
            sample['cloth'].faces_batch: torch.LongTensor [3xF]
            ample['cloth', 'mesh_edge', 'cloth'].edge_index: torch.LongTensor [2xE]
        """

        garment_dict = self.garments_dict[garment_name]

        faces = torch.tensor(garment_dict['faces'])
        edges = triangles_to_edges(faces.unsqueeze(0))
        sample['cloth', 'mesh_edge', 'cloth'].edge_index = edges

        sample['cloth'].faces_batch = faces.T

        return sample

    def make_vertex_level(self, sample, coarse_edges_dict: Dict[int, np.array]):
        """
        Add `vertex_level` labels to `sample['cloth']`
        for each garment vertex, `vertex_level` is the number of the deepest level the vertex is in
        starting from `0` for the most shallow level

        :param sample: HeteroData
        :param coarse_edges_dict: dictionary with list of edges for each coarse level
        :return: sample['cloth'].vertex_level: torch.LongTensor [Vx1]
        """
        N = sample['cloth'].pos.shape[0]
        vertex_level = np.zeros((N, 1)).astype(np.int64)
        for i in range(self.mcfg.n_coarse_levels):
            edges_coarse = coarse_edges_dict[i].astype(np.int64)
            nodes_unique = np.unique(edges_coarse.reshape(-1))
            vertex_level[nodes_unique] = i + 1
        sample['cloth'].vertex_level = torch.tensor(vertex_level)
        return sample

    def add_coarse(self, sample, garment_name):
        """
        Add coarse edges to `sample` as `sample['cloth', f'coarse_edge{i}', 'cloth'].edge_index`.
        where `i` is the number of the coarse level (starting from `0`)

        :param sample: HeteroData
        :param garment_name:
        :return: sample['cloth', f'coarse_edge{i}', 'cloth'].edge_index: torch.LongTensor [2, E_i]
        """
        if self.mcfg.n_coarse_levels == 0:
            return sample

        garment_dict = self.garments_dict[garment_name]
        faces = garment_dict['faces']

        # Randomly choose center of the mesh
        # center of a graph is a node with minimal eccentricity (distance to the farthest node)
        center_nodes = garment_dict['center']
        center = center_nodes[0]

        if 'coarse_edges' not in garment_dict:
            garment_dict['coarse_edges'] = dict()

        # if coarse edges are already precomputed for the given `center`,
        # take them from `garment_dict['coarse_edges'][center]`
        # else compute them with `make_coarse_edges` and stash in `garment_dict['coarse_edges'][center]`
        if center in garment_dict['coarse_edges']:
            coarse_edges_dict = garment_dict['coarse_edges'][center]
        else:
            coarse_edges_dict = make_coarse_edges(faces, center, n_levels=self.mcfg.n_coarse_levels)
            garment_dict['coarse_edges'][center] = coarse_edges_dict

        # for each level `i` add edges to sample as  `sample['cloth', f'coarse_edge{i}', 'cloth'].edge_index`
        for i in range(self.mcfg.n_coarse_levels):
            key = f'coarse_edge{i}'
            edges_coarse = coarse_edges_dict[i].astype(np.int64)
            edges_coarse = np.concatenate([edges_coarse, edges_coarse[:, [1, 0]]], axis=0)
            coarse_edges = torch.tensor(edges_coarse.T)
            sample['cloth', key, 'cloth'].edge_index = coarse_edges

        # add `vertex_level` labels to sample
        sample = self.make_vertex_level(sample, coarse_edges_dict)

        return sample
    
    def add_button_edges(self, sample, garment_name: str):
        """
        Add set of node pairs that should serve as buttons (needed for unzipping/unbuttoning demonstration)
        :param sample: HeteroData
        :param garment_name: name of the garment in `self.garment_smpl_model_dict` and `self.garments_dict`
        :return: if `button_edges` are on,
            sample['cloth'].button_edges_batch: torch.LongTensor [2xBE]
        """

        # if button_edges flag is off, do nothing
        if not hasattr(self.mcfg, 'button_edges') or not self.mcfg.button_edges:
            return sample

        garment_dict = self.garments_dict[garment_name]

        # if there are no buttons for the given garment, do nothing
        if 'button_edges' not in garment_dict:
            return sample

        button_edges = garment_dict['button_edges']


        button_edges = torch.LongTensor(button_edges).T
        sample['cloth', 'mesh_edge', 'cloth'].edge_index = torch.cat([sample['cloth', 'mesh_edge', 'cloth'].edge_index,button_edges], dim=1)

        return sample
    
    def add_garment_id(self, sample, garment_name: str):

        garment_dict = self.garments_dict[garment_name]
        if 'garment_id' not in garment_dict:
            return sample
        
        garment_id = garment_dict['garment_id']
        sample['cloth'].garment_id = torch.LongTensor(garment_id)
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


    def add_garment_to_sample(self, sample_full, sample_garment):

        if 'cloth' in sample_full.node_types:
            cloth_data_full = sample_full['cloth']
            N_exist = sample_full['cloth'].pos.shape[0]
            garment_id = sample_full['cloth'].garment_id.max()+1
        else:
            cloth_data_full = None
            N_exist = 0
            garment_id = 0



        cloth_data_garment = sample_garment['cloth']
        N_new = cloth_data_garment.pos.shape[0]


        if 'garment_id' in sample_garment['cloth']:
            garment_id_tensor = sample_garment['cloth'].garment_id + garment_id
        else:
            garment_id_tensor = torch.ones(N_new, 1, dtype=torch.long) * garment_id
        cloth_data_garment.garment_id = garment_id_tensor

        for edge_type in sample_garment.edge_types:
            edge_data_new = sample_garment[edge_type]
            if edge_type in sample_full.edge_types:
                edge_data_existing = sample_full[edge_type]
            else:
                edge_data_existing = None

            for k, v in edge_data_new._mapping.items():
                v += N_exist
                if edge_data_existing is not None:
                    dim_cat = 1 if 'index' in k else 0
                    v = torch.cat([edge_data_existing[k], v], dim=dim_cat)
                sample_full[edge_type][k] = v
            torch_geometric.data.storage.EdgeStorage

        for k, v in cloth_data_garment._mapping.items():
            if 'batch' in k:
                v += N_exist
            if cloth_data_full is not None:
                dim_cat = 1 if 'batch' in k else 0
                v = torch.cat([cloth_data_full[k], v], dim=dim_cat)
            sample_full['cloth'][k] = v
        return sample_full
    
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



class BodyBuilder:
    def __init__(self, mcfg, smpl_models_dict, obstacle_dict):
        self.smpl_models_dict = smpl_models_dict
        self.obstacle_dict = obstacle_dict
        self.mcfg = mcfg
        self.vertex_builder = VertexBuilder(mcfg)



    def make_smpl_vertices(self, smplx_dict, gender, **kwargs):
        full_pose = torch.FloatTensor(smplx_dict['body_pose'])
        body_pose = torch.FloatTensor(smplx_dict['body_pose'])
        global_orient = torch.FloatTensor(smplx_dict['global_orient'])
        transl = torch.FloatTensor(smplx_dict['transl'])
        betas = torch.FloatTensor(smplx_dict['betas'])
        expression = torch.FloatTensor(smplx_dict['expression'])
        jaw_pose = torch.FloatTensor(smplx_dict['jaw_pose'])
        leye_pose = torch.FloatTensor(smplx_dict['leye_pose'])
        reye_pose = torch.FloatTensor(smplx_dict['reye_pose'])
        left_hand_pose = torch.FloatTensor(smplx_dict['left_hand_pose'])
        right_hand_pose = torch.FloatTensor(smplx_dict['right_hand_pose'])

        smplx_dict_new = {}
        smplx_dict_new['betas'] = betas
        smplx_dict_new['expression'] = expression
        smplx_dict_new['global_orient'] = global_orient
        smplx_dict_new['transl'] = transl
        smplx_dict_new['body_pose'] = full_pose[:, :63]
        smplx_dict_new['jaw_pose'] = jaw_pose
        smplx_dict_new['leye_pose'] = leye_pose
        smplx_dict_new['reye_pose'] = reye_pose
        smplx_dict_new['left_hand_pose'] = left_hand_pose
        smplx_dict_new['right_hand_pose'] = right_hand_pose


        if len(body_pose.shape) == 1:
            body_pose = body_pose.unsqueeze(0)
            global_orient = global_orient.unsqueeze(0)
            transl = transl.unsqueeze(0)
        if len(betas.shape) == 1:
            betas = betas.unsqueeze(0)

        smpl_model = self.smpl_models_dict[gender]

        with torch.no_grad():
            smpl_output = smpl_model(**smplx_dict_new)
        vertices = smpl_output.vertices.numpy().astype(np.float32)

        if smplx_dict['type'] == 'amass':

            r_permute = np.array([[1,0,0],
                        [0,0,1],
                        [0,1,0]], dtype=vertices.dtype)
            
            vertices = vertices @ r_permute

        return vertices
    
    def make_vertex_type(self, n_verts):
        vertex_type = np.ones((n_verts, 1)).astype(np.int64)

        if not self.mcfg.omit_hands or self.obstacle_dict is None:
            return vertex_type

        for k in ['rightHand', 'leftHand', 'rightHandIndex1', 'leftHandIndex1']:
            node_ids = self.obstacle_dict[k]
            vertex_type[node_ids] = 2

        return vertex_type


    def add_vertex_type(self, sample):
        N = sample['obstacle'].pos.shape[0]
        vertex_type = self.make_vertex_type(N)
        sample['obstacle'].vertex_type = torch.tensor(vertex_type)
        return sample

    def add_faces(self, sample, gender):
        smpl_model = self.smpl_models_dict[gender]
        faces = torch.tensor(smpl_model.faces.astype(np.int64))
        sample['obstacle'].faces_batch = faces.T
        return sample

    def add_vertex_level(self, sample):
        if self.mcfg.n_coarse_levels == 0:
            return sample
        N = sample['obstacle'].pos.shape[0]
        vertex_level = torch.zeros(N, 1).long()
        sample['obstacle'].vertex_level = vertex_level
        return sample

    def build(self, sample, sequence_dict, gender):


        make_smpl_vertices = partial(self.make_smpl_vertices, gender=gender)
        sample, _ = self.vertex_builder.add_verts(sample, sequence_dict, make_smpl_vertices, "obstacle")
        sample = self.add_vertex_type(sample)
        sample = self.add_faces(sample, gender)
        sample = self.add_vertex_level(sample)
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
    def __init__(self, mcfg, garments_dict, smpl_models_dict, obstacle_dict):

        self.sequence_loader = PKLFolderSequenceLoader(mcfg)
        self.garment_builder = GarmentBuilder(mcfg, garments_dict)
        self.body_builder = BodyBuilder(mcfg, smpl_models_dict, obstacle_dict)
        self.mcfg = mcfg

        self.registration_root = mcfg.registration_root
        self.seq_root = mcfg.data_root

    def load_sample(self, subject, garment_name, reg_gname, take, gender):

        smplx_sequence_path = os.path.join(self.seq_root, subject, garment_name, take, 'Meshes', 'smplx')
        smpl_sequence = self.sequence_loader.load_sequence(smplx_sequence_path)

        sample = HeteroData()
        sample = self.body_builder.build(sample, smpl_sequence, gender)


        garment_seq_dir = os.path.join(self.registration_root, reg_gname, take.lower(), 'meshes')
        sample = self.garment_builder.build(sample, garment_seq_dir, reg_gname, smpl_sequence)



        return sample


class Dataset:
    def __init__(self, loader: Loader, datasplit: pd.DataFrame):
        self.loader = loader
        self.datasplit = datasplit
        self._len = self.datasplit.shape[0]

        self.stash = {}


    def __getitem__(self, item):
        row = self.datasplit.iloc[item]

        subject = row.subject
        garment_name = row.garment
        reg_gname = row.reg_gname
        take = row['take']
        gender = row.gender


        sequence_name = f"{reg_gname}_{take}"
        if sequence_name in self.stash:
            return self.stash[sequence_name]
        else:

            sample = self.loader.load_sample(subject, garment_name, reg_gname, take, gender)
            sample['sequence_name'] = sequence_name
            sample['garment_name'] = reg_gname

            self.stash[sequence_name] = sample


        return sample

    def __len__(self):
        return self._len