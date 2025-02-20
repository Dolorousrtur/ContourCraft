# individual garment dicts
from functools import partial
import importlib
import os
import pickle
from dataclasses import dataclass, MISSING
from random import random
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import smplx
import torch
from smplx import SMPL
from torch_geometric.data import HeteroData

from utils.coarse import make_coarse_edges
from utils.common import NodeType, triangles_to_edges, separate_arms
from utils.datasets import build_smpl_bygender, convert_lbs_dict, load_garments_dict, make_garment_smpl_dict
from utils.defaults import DEFAULTS
from utils.garment_smpl import GarmentSMPL
from utils.io import pickle_load


@dataclass
class Config:
    # garment_dict_file: str = MISSING  # Path to the garment dict file with data for all garments relative to $HOOD_DATA/aux_data/
    garment_dicts_dir: str = MISSING  # Path to the garment dict file with data for all garments relative to $HOOD_DATA/aux_data/
    data_root: str = MISSING  # Path to the data root relative to $HOOD_DATA/
    body_model_root: str = 'body_models'  # Path to the directory containg body model files, should contain `smpl` and/or `smplx` sub-directories. Relative to $HOOD_DATA/aux_data/
    model_type: str = 'smpl'  # Type of the body model ('smpl' or 'smplx')
    split_path: Optional[str] = None  # Path to the .csv split file relative to $HOOD_DATA/aux_data/
    obstacle_dict_file: Optional[
        str] = None  # Path to the file with auxiliary data for obstacles relative to $HOOD_DATA/aux_data/
    
    # rollout_steps: int = -1

    sequence_loader: str = 'hood_pkl'  # Name of the sequence loader to use 
    noise_scale: float = 3e-3  # Noise scale for the garment vertices (not used in validation)
    lookup_steps: int = 5  # Number of steps to look up in the future (not used in validation)
    pinned_verts: bool = False  # Whether to use pinned vertices
    wholeseq: bool = False  # whether to load the whole sequence (always True in validation)
    random_betas: bool = False  # Whether to use random beta parameters for the SMPL model
    use_betas_for_restpos: bool = False  # Whether to use beta parameters to get canonical garment geometry
    betas_scale: float = 0.1  # Scale for the beta parameters (not used if random_betas is False)
    restpos_scale_min: float = 1.  # Minimum scale for randomly sampling the canonical garment geometry
    restpos_scale_max: float = 1.  # Maximum scale for randomly sampling the canonical garment geometry
    n_coarse_levels: int = 1  # Number of coarse levels with long-range edges
    separate_arms: bool = False  # Whether to separate the arms from the rest of the body (to avoid body self-intersections)
    zero_betas: bool = False  # Whether to set the beta parameters to zero
    button_edges: bool = False  # Whether to load the button edges
    single_sequence_file: Optional[str] = None  # Path to the single sequence to load (used in Inference.ipynb)
    single_sequence_garment: Optional[
        str] = None  # Garment name for the single sequence to load (used in Inference.ipynb)
    gender: Optional[str] = None  # Gender of the body model, only used for one-sequence inference

    betas_file: Optional[
        str] = None  # Path to the file with the table of beta parameters (used in validation to generate sequences with specific body shapes)

    nobody_freq: float = 0.
    fps: int = 30  # Target FPS for the sequence

def make_obstacle_dict(mcfg: Config) -> dict:
    if mcfg.obstacle_dict_file is None:
        return {}

    obstacle_dict_path = os.path.join(DEFAULTS.aux_data, mcfg.obstacle_dict_file)
    with open(obstacle_dict_path, 'rb') as f:
        obstacle_dict = pickle.load(f)
    return obstacle_dict


def create_loader(mcfg: Config):
    # garment_dict_path = os.path.join(DEFAULTS.aux_data, mcfg.garment_dict_file)
    garment_dict_dir = os.path.join(DEFAULTS.aux_data, mcfg.garment_dicts_dir)

    # garments_dict = load_garments_dict(garment_dict_path)

    body_model_root = os.path.join(DEFAULTS.aux_data, mcfg.body_model_root)

    if mcfg.sequence_loader == 'hood_pkl':
        mcfg.model_type = 'smpl'
    elif mcfg.sequence_loader == 'cmu_npz_smpl':
        mcfg.model_type = 'smpl'
    elif mcfg.sequence_loader == 'cmu_npz_smplx':
        mcfg.model_type = 'smplx'

    # body_model = smplx.create(body_model_root, model_type=mcfg.model_type, gender=mcfg.gender, use_pca=False)
    body_models_dict = build_smpl_bygender(body_model_root, mcfg.model_type)

    # garment_smpl_model_dict = make_garment_smpl_dict(garments_dict, body_model)
    obstacle_dict = make_obstacle_dict(mcfg)

    if mcfg.single_sequence_file is None:
        mcfg.data_root = os.path.join(DEFAULTS.CMU_root, mcfg.data_root)

    if mcfg.betas_file is not None:
        betas_table = pickle_load(os.path.join(DEFAULTS.aux_data, mcfg.betas_file))['betas']
    else:
        betas_table = None

    loader = Loader(mcfg, 
                    body_models_dict, garment_dict_dir, obstacle_dict=obstacle_dict, betas_table=betas_table)
    return loader


def create(mcfg: Config):
    loader = create_loader(mcfg)

    if mcfg.single_sequence_file is not None:
        datasplit = pd.DataFrame()
        datasplit['id'] = [mcfg.single_sequence_file]
        datasplit['garment'] = [mcfg.single_sequence_garment]
        datasplit['gender'] = [mcfg.gender]
    else:
        split_path = os.path.join(DEFAULTS.aux_data, mcfg.split_path)
        datasplit = pd.read_csv(split_path, dtype='str')

    dataset = Dataset(loader, datasplit, wholeseq=mcfg.wholeseq)
    return dataset


class VertexBuilder:
    """
    Helper class to build garment and body vertices from a sequence of SMPL poses.
    """

    def __init__(self, mcfg):
        self.mcfg = mcfg

    @staticmethod
    def build(sequence_dict: dict, f_make, idx_start: int, idx_end: int = None, garment_name: str = None) -> np.ndarray:
        """
        Build vertices from a sequence of SMPL poses using the given `f_make` function.
        :param sequence_dict: a dictionary of SMPL parameters
        :param f_make: a function that takes SMPL parameters and returns vertices
        :param idx_start: first frame index
        :param idx_end: last frame index
        :param garment_name: name of the garment (None for body)
        :return: [Nx3] mesh vertices
        """

        input_dict = {}
        n_frames = sequence_dict['body_pose'].shape[0]

        for k in ['betas', 'expression']:
            if k in sequence_dict:
                v = sequence_dict[k]
                if len(v.shape) == 1:
                    v = v[None]

                if len(v.shape) == 2 and v.shape[0] == 1:
                    v = v.repeat(n_frames, 0)

                if len(v.shape) == 2 and v.shape[0] != 1:
                    v = v[idx_start: idx_end]
                input_dict[k] = v

        for k in ['global_orient', 'body_pose', 'transl', 
            'left_hand_pose', 'right_hand_pose', 
            'jaw_pose', 'leye_pose', 'reye_pose']: 
            if k in sequence_dict:
                input_dict[k] = sequence_dict[k][idx_start: idx_end]

                

        verts = f_make(input_dict, garment_name=garment_name)


        return verts

    def pad_verts(self, vertices: np.ndarray, n_steps) -> np.ndarray:
        """
        Pad the vertex sequence to the required number of steps.
        """
        n_lookup = vertices.shape[0]
        n_topad = n_steps - n_lookup
        # n_topad = self.mcfg.lookup_steps - n_lookup

        if n_topad == 0:
            return vertices

        padlist = [vertices] + [vertices[-1:]] * n_topad
        vertices = np.concatenate(padlist, axis=0)
        return vertices

    def pos2tensor(self, pos: np.ndarray) -> torch.Tensor:
        """
        Convert a numpy array of vertices to a tensor and permute the axes into [VxNx3] (torch geometric format)
        """

        pos = torch.tensor(pos)

        if len(pos.shape) == 3:
            pos= pos.permute(1, 0, 2)

        # print('self.mcfg.wholeseq', self.mcfg.wholeseq)
        # print('pos.shape', pos.shape)

        # if not self.mcfg.wholeseq and pos.shape[1] == 1:
        #     pos = pos[:, 0]
        return pos
    
    def permute_axes(self, vertices: np.ndarray) -> np.ndarray:  
        if self.mcfg.sequence_loader in ['cmu_npz_smpl', 'cmu_npz_smplx']:

            r_permute = np.array([[1,0,0],
                        [0,0,-1],
                        [0,1,0]], dtype=vertices.dtype)
            

            vertices = vertices @ r_permute

        return vertices

    def add_verts(self, sample: HeteroData, sequence_dict: dict, idx: int, f_make, object_key: str,
                  **kwargs) -> HeteroData:
        """
        Builds the vertices from the given SMPL pose sequence and adds them to the HeteroData sample.
        :param sample: HetereoData object
        :param sequence_dict: sequence of SMPL parameters
        :param idx: frame index (not used if self.mcfg.wholeseq is True)
        :param f_make: function that takes SMPL parameters and returns vertices
        :param object_key: name of the object to build vertices for ('cloth' or 'obstacle')
        :return: updated HeteroData object
        """

        N_steps = sequence_dict['body_pose'].shape[0]
        pos_dict = {}

        # Build the vertices for the whole sequence
        if self.mcfg.wholeseq:
            all_vertices = VertexBuilder.build(sequence_dict, f_make, 0, None,
                                               **kwargs)
            all_vertices = self.permute_axes(all_vertices)
            # pos_dict['prev_pos'] = all_vertices[:-2]
            # pos_dict['pos'] = all_vertices[1:-1]
            # pos_dict['target_pos'] = all_vertices[2:]

            pos_dict["prev_pos"] = all_vertices[0]
            pos_dict["pos"] = all_vertices[1]
            pos_dict["target_pos"] = all_vertices[2]

            lookup = all_vertices[2:]
            # lookup = self.pad_lookup(lookup)

            pos_dict["lookup"] = lookup

        # Build the vertices for several frames starting from `idx`
        else:
            n_lookup = min(self.mcfg.lookup_steps, N_steps - idx - 2)
            all_vertices = VertexBuilder.build(sequence_dict, f_make, idx, idx + 2 + n_lookup,
                                               **kwargs)
            all_vertices = self.permute_axes(all_vertices)
            # pos_dict["prev_pos"] = all_vertices[:1]
            # pos_dict["pos"] = all_vertices[1:2]
            # pos_dict["target_pos"] = all_vertices[2:3]


            all_vertices = self.pad_verts(all_vertices, self.mcfg.lookup_steps+3)

            pos_dict["prev_pos"] = all_vertices[0]
            pos_dict["pos"] = all_vertices[1]
            pos_dict["target_pos"] = all_vertices[2]

            lookup = all_vertices[2:]

            pos_dict["lookup"] = lookup

        for k, v in pos_dict.items():
            v = self.pos2tensor(v)
            setattr(sample[object_key], k, v)

        return sample


class NoiseMaker:
    """
    Helper class to add noise to the garment vertices.
    """

    def __init__(self, mcfg: Config):
        self.mcfg = mcfg

    def add_noise(self, sample: HeteroData) -> HeteroData:
        """
        Add gaussian noise with std == self.mcfg.noise_scale to `pos` and `prev_pos`
        tensors in `sample['cloth']`
        :param sample: HeteroData
        :return: sample: HeteroData
        """
        if self.mcfg.noise_scale == 0:
            return sample

        world_pos = sample['cloth'].pos
        vertex_type = sample['cloth'].vertex_type
        if len(vertex_type.shape) == 1:
            vertex_type = vertex_type[..., None]

        noise = np.random.normal(scale=self.mcfg.noise_scale, size=world_pos.shape).astype(np.float32)
        noise_prev = np.random.normal(scale=self.mcfg.noise_scale, size=world_pos.shape).astype(np.float32)

        noise = torch.tensor(noise)
        noise_prev = torch.tensor(noise_prev)

        mask = vertex_type == NodeType.NORMAL
        if len(mask.shape) == 2 and len(noise.shape) == 3:
            mask = mask.unsqueeze(-1)
        noise = noise * mask

        sample['cloth'].pos = sample['cloth'].pos + noise
        sample['cloth'].prev_pos = sample['cloth'].prev_pos + noise_prev

        return sample


class GarmentBuilder:
    """
    Class to build the garment meshes from SMPL parameters.
    """

    # def __init__(self, mcfg: Config, garments_dict: dict, garment_smpl_model_dict: Dict[str, GarmentSMPL]):
    def __init__(self, mcfg: Config, body_models_dict, garment_dicts_dir: str):
        """
        :param mcfg: config
        :param garments_dict: dictionary with data for all garments
        :param garment_smpl_model_dict: dictionary with SMPL models for all garments
        """
        self.mcfg = mcfg
        self.garment_dicts_dir = garment_dicts_dir
        self.garments_dict = {}
        self.garment_smpl_model_dict = {}
        # self.body_model = body_model
        self.body_models_dict = body_models_dict

        self.vertex_builder = VertexBuilder(mcfg)
        self.noise_maker = NoiseMaker(mcfg)


    
    def make_cloth_verts(self, sequence_dict, garment_name: str) -> np.ndarray:
        """
        Create body vertices from SMPL parameters (used in VertexBuilder.add_verts)

        :param body_pose: SMPL pose parameters [Nx69] OR [69]
        :param global_orient: SMPL global_orient [Nx3] OR [3]
        :param transl: SMPL translation [Nx3] OR [3]
        :param betas: SMPL betas [Nx10] OR [10]

        :return: vertices [NxVx3]
        """
        # body_pose = torch.FloatTensor(body_pose)
        # global_orient = torch.FloatTensor(global_orient)
        # transl = torch.FloatTensor(transl)
        # betas = torch.FloatTensor(betas)

        sequence_dict = {k:torch.FloatTensor(v) for k,v in sequence_dict.items()}

        if len(sequence_dict['body_pose'].shape) == 1:
            for k in ['global_orient', 'body_pose', 'transl', 
            'left_hand_pose', 'right_hand_pose', 
            'jaw_pose', 'leye_pose', 'reye_pose']: 
                if k in sequence_dict:
                    sequence_dict[k] = sequence_dict[k].unsqueeze(0)
        wholeseq = self.mcfg.wholeseq or sequence_dict['body_pose'].shape[0] > 1



        betas = sequence_dict['betas']
        if len(betas.shape) == 1:
            betas = betas.unsqueeze(0)

        global_orient = sequence_dict['global_orient']
        body_pose = sequence_dict['body_pose']
        transl = sequence_dict['transl']
        full_pose = []
        for k in ['global_orient', 'body_pose', 
        'left_hand_pose', 'right_hand_pose', 
        'jaw_pose', 'leye_pose', 'reye_pose']: 
            if k in sequence_dict:
                full_pose.append(sequence_dict[k])

        full_pose = torch.cat(full_pose, dim=1)



        garment_smpl_model = self.garment_smpl_model_dict[garment_name]
        with torch.no_grad():
            vertices = garment_smpl_model.make_vertices(betas=betas, full_pose=full_pose, transl=transl).numpy()

        if not wholeseq:
            vertices = vertices[0]

        return vertices



    def add_vertex_type(self, sample: HeteroData, garment_name: str) -> HeteroData:
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

        if self.mcfg.pinned_verts:
            vertex_type = garment_dict['node_type'].astype(np.int64)
        else:
            V = sample['cloth'].pos.shape[0]
            vertex_type = np.zeros((V, 1)).astype(np.int64)

        sample['cloth'].vertex_type = torch.tensor(vertex_type)
        return sample

    def resize_restpos(self, restpos: np.array) -> np.array:
        """
        Randomly resize resting geometry of a garment
        with scale from `self.mcfg.restpos_scale_min` to `self.mcfg.restpos_scale_max`

        :param restpos: Vx3
        :return: resized restpos: Vx3
        """
        if self.mcfg.restpos_scale_min == self.mcfg.restpos_scale_max == 1.:
            return restpos

        scale = np.random.rand()
        scale *= (self.mcfg.restpos_scale_max - self.mcfg.restpos_scale_min)
        scale += self.mcfg.restpos_scale_min

        mean = restpos.mean(axis=0, keepdims=True)
        restpos -= mean
        restpos *= scale
        restpos += mean

        return restpos

    def make_shaped_restpos(self, sequence_dict: dict, garment_name: str) -> np.ndarray:
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
        body_pose = np.zeros_like(sequence_dict['body_pose'][:1])
        global_orient = np.zeros_like(sequence_dict['global_orient'][:1])
        transl = np.zeros_like(sequence_dict['transl'][:1])
        verts = self.make_cloth_verts(body_pose,
                                      global_orient,
                                      transl,
                                      sequence_dict['betas'], garment_name=garment_name)
        return verts

    def add_restpos(self, sample: HeteroData, sequence_dict: dict, garment_name: str) -> HeteroData:
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
            rest_pos = self.resize_restpos(garment_dict['rest_pos'])

        sample['cloth'].rest_pos = torch.tensor(rest_pos).float()
        return sample


    def add_faces_and_edges(self, sample: HeteroData, garment_name: str) -> HeteroData:
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

    def make_vertex_level(self, sample: HeteroData, coarse_edges_dict: Dict[int, np.array]) -> HeteroData:
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

    def add_coarse(self, sample: HeteroData, garment_name: str) -> HeteroData:
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
        center = np.random.choice(center_nodes)
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

    def add_button_edges(self, sample: HeteroData, garment_name: str) -> HeteroData:
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

        button_edges = torch.LongTensor(button_edges)
        sample['cloth'].button_edges_batch = button_edges.T

        return sample
    
        
    def add_garment_id(self, sample, garment_name: str):

        garment_dict = self.garments_dict[garment_name]
        if 'garment_id' not in garment_dict:
            return sample
        
        garment_id = garment_dict['garment_id']
        n_garments = np.unique(garment_id).shape[0]
        if n_garments == 1:
            return sample

        sample['cloth'].garment_id = torch.LongTensor(garment_id)
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


        for k, v in cloth_data_garment._mapping.items():
            if 'batch' in k:
                v += N_exist
            if cloth_data_full is not None:
                dim_cat = 1 if 'batch' in k else 0
                v = torch.cat([cloth_data_full[k], v], dim=dim_cat)
            sample_full['cloth'][k] = v
        return sample_full

    def load_garment_dict(self, garment_name: str) -> dict:
        if garment_name not in self.garments_dict:
            garment_dict_path = os.path.join(self.garment_dicts_dir, garment_name + '.pkl')
            garment_dict = pickle_load(garment_dict_path)

            garment_dict['lbs'] = convert_lbs_dict(garment_dict['lbs'])
            self.garments_dict[garment_name] = garment_dict

        if garment_name not in self.garment_smpl_model_dict:
            garment_dict = self.garments_dict[garment_name]
            gender = garment_dict['gender']
            body_model = self.body_models_dict[gender]
            garment_smpl_model = GarmentSMPL(body_model, garment_dict['lbs'])
            self.garment_smpl_model_dict[garment_name] = garment_smpl_model

    def build(self, sample: HeteroData, sequence_dict: dict, idx: int, garment_name: str) -> HeteroData:
        """
        Add all data for the garment to the sample

        :param sample: HeteroData
        :param sequence_dict: dictionary with SMPL parameters
        :param idx: starting index in a sequence (not used if  `self.mcfg.wholeseq`)
        :param garment_name: name of the garment in `self.garment_smpl_model_dict` and `self.garments_dict`
        :return:
            if self.mcfg.wholeseq:
                sample['cloth'].prev_pos torch.FloatTensor [VxNx3]: vertex positions at the previous frame
                sample['cloth'].pos torch.FloatTensor [VxNx3]: vertex positions at the current frame
                sample['cloth'].target_pos torch.FloatTensor [VxNx3]: vertex positions at the next frame
            else:
                sample['cloth'].prev_pos torch.FloatTensor [Vx3]: vertex positions at the previous frame
                sample['cloth'].pos torch.FloatTensor [Vx3]: vertex positions at the current frame
                sample['cloth'].target_pos torch.FloatTensor [Vx3]: vertex positions at the next frame
                sample['cloth'].lookup torch.FloatTensor [VxLx3] (L == self.mcfg.lookup_steps): vertex positions at several future frames

            sample['cloth'].rest_pos torch.FloatTensor [Vx3]: vertex positions in the canonical pose
            sample['cloth'].faces_batch torch.LongTensor [3xF]: garment faces
            sample['cloth'].vertex_type torch.LongTensor [Vx1]: vertex type (0 - regular, 3 - pinned)
            sample['obstacle'].vertex_level torch.LongTensor [Vx1]: level of the vertex in the hierarchy (always 0 for the body)

            sample['cloth', 'mesh_edge', 'cloth'].edge_index: torch.LongTensor [2xE]: mesh edges

            for each coarse level `i` in [0, self.mcfg.n_coarse_levels]:
                sample['cloth', f'coarse_edge{i}', 'cloth'].edge_index: torch.LongTensor [2, E_i]: coarse edges at level `i`

            if self.mcfg.button edges and the garment has buttons:
                sample['cloth'].button_edges_batch: torch.LongTensor [2xBE]: button edges

        """

        self.load_garment_dict(garment_name)

        sample_garment = HeteroData()
        sample_garment = self.vertex_builder.add_verts(sample_garment, sequence_dict, idx, self.make_cloth_verts, "cloth",
                                               garment_name=garment_name)

        sample_garment = self.add_vertex_type(sample_garment, garment_name)
        sample_garment = self.noise_maker.add_noise(sample_garment)
        sample_garment = self.add_restpos(sample_garment, sequence_dict, garment_name)
        sample_garment = self.add_faces_and_edges(sample_garment, garment_name)
        sample_garment = self.add_coarse(sample_garment, garment_name)
        sample_garment = self.add_button_edges(sample_garment, garment_name)
        sample_garment = self.add_garment_id(sample_garment, garment_name)

        sample = self.add_garment_to_sample(sample, sample_garment)

        return sample


class BodyBuilder:
    """
    Class for building body meshed from SMPL parameters
    """

    def __init__(self, mcfg: Config, body_models_dict: SMPL, obstacle_dict: dict):
        """
        :param mcfg: Config
        :param smpl_model:
        :param obstacle_dict: auxiliary data for the obstacle
                obstacle_dict['vertex_type']: vertex type (1 - regular obstacle node, 2 - hand node (omitted during inference to avoid body self-penetrations))
        """
        # self.smpl_model = smpl_model
        self.body_models_dict = body_models_dict
        self.obstacle_dict = obstacle_dict
        self.mcfg = mcfg
        self.vertex_builder = VertexBuilder(mcfg)

    def make_smpl_vertices(self, sequence_dict, gender, **kwargs) -> np.ndarray:
        """
        Create body vertices from SMPL parameters (used in VertexBuilder.add_verts)

        :param body_pose: SMPL pose parameters [Nx69] OR [69]
        :param global_orient: SMPL global_orient [Nx3] OR [3]
        :param transl: SMPL translation [Nx3] OR [3]
        :param betas: SMPL betas [Nx10] OR [10]

        :return: vertices [NxVx3]
        """

        sequence_dict = {k:torch.FloatTensor(v) for k,v in sequence_dict.items()}

        if len(sequence_dict['body_pose'].shape) == 1:
            for k in ['global_orient', 'body_pose', 'transl', 
            'left_hand_pose', 'right_hand_pose', 
            'jaw_pose', 'leye_pose', 'reye_pose']: 
                if k in sequence_dict:
                    sequence_dict[k] = sequence_dict[k].unsqueeze(0)

        for k in ['betas', 'expression']:
            if k in sequence_dict:
                v = sequence_dict[k]
                if len(v.shape) == 1:
                    v = v.unsqueeze(0)
                sequence_dict[k] = v

        wholeseq = self.mcfg.wholeseq or sequence_dict['body_pose'].shape[0] > 1

        body_model = self.body_models_dict[gender]
        with torch.no_grad():
            smpl_output = body_model(**sequence_dict)
        vertices = smpl_output.vertices.numpy().astype(np.float32)

        if not wholeseq:
            vertices = vertices[0]

        return vertices

    def add_vertex_type(self, sample: HeteroData) -> HeteroData:
        """
        Add vertex type field to the obstacle object in the sample
        """
        N = sample['obstacle'].pos.shape[0]
        if 'vertex_type' in self.obstacle_dict:
            vertex_type = self.obstacle_dict['vertex_type']
        else:
            vertex_type = np.ones((N, 1)).astype(np.int64)
        sample['obstacle'].vertex_type = torch.tensor(vertex_type)
        return sample

    def add_faces(self, sample: HeteroData, gender:str) -> HeteroData:
        """
        Add body faces to the obstacle object in the sample
        """
        body_model = self.body_models_dict[gender]
        faces = torch.tensor(body_model.faces.astype(np.int64))
        sample['obstacle'].faces_batch = faces.T
        return sample

    def add_vertex_level(self, sample: HeteroData) -> HeteroData:
        """
        Add vertex level field to the obstacle object in the sample (always 0 for the body)
        """
        N = sample['obstacle'].pos.shape[0]
        vertex_level = torch.zeros(N, 1).long()
        sample['obstacle'].vertex_level = vertex_level
        return sample

    def build(self, sample: HeteroData, sequence_dict: dict, idx: int, gender: str) -> HeteroData:
        """
        Add all data for the body (obstacle) to the sample
        :param sample: HeteroData object to add data to
        :param sequence_dict: dict with SMPL parameters
        :param idx: index of the current frame in the sequence
        
        :return:
            if self.mcfg.wholeseq:
                sample['obstacle'].prev_pos torch.FloatTensor [VxNx3]: vertex positions at the previous frame
                sample['obstacle'].pos torch.FloatTensor [VxNx3]: vertex positions at the current frame
                sample['obstacle'].target_pos torch.FloatTensor [VxNx3]: vertex positions at the next frame
            else:
                sample['obstacle'].prev_pos torch.FloatTensor [Vx3]: vertex positions at the previous frame
                sample['obstacle'].pos torch.FloatTensor [Vx3]: vertex positions at the current frame
                sample['obstacle'].target_pos torch.FloatTensor [Vx3]: vertex positions at the next frame
                sample['obstacle'].lookup torch.FloatTensor [VxLx3] (L == self.mcfg.lookup_steps): vertex positions at several future frames

            sample['obstacle'].faces_batch torch.LongTensor [3xF]: garment faces
            sample['obstacle'].vertex_type torch.LongTensor [Vx1]: vertex type (1 - regular obstacle, 2 - omitted)
            sample['obstacle'].vertex_level torch.LongTensor [Vx1]: level of the vertex in the hierarchy (always 0 for the body)

        """

        make_smpl_vertices = partial(self.make_smpl_vertices, gender=gender)
        sample = self.vertex_builder.add_verts(sample, sequence_dict, idx, make_smpl_vertices, "obstacle")
        sample = self.add_vertex_type(sample)
        sample = self.add_faces(sample, gender=gender)
        sample = self.add_vertex_level(sample)
        return sample




class Loader:
    """
    Class for building HeteroData objects containing all data for a single sample
    """

    # def __init__(self, mcfg: Config, garments_dict: dict, smpl_model: SMPL,
                #  garment_smpl_model_dict: Dict[str, GarmentSMPL], obstacle_dict: dict, betas_table=None):
    def __init__(self, mcfg: Config, body_models_dict: dict, garment_dicts_dir: str, obstacle_dict: dict, betas_table=None):
        
        sequence_loader_module = importlib.import_module(f'datasets.sequence_loaders.{mcfg.sequence_loader}')
        SequenceLoader = sequence_loader_module.SequenceLoader

        self.sequence_loader = SequenceLoader(mcfg, mcfg.data_root, betas_table=betas_table)
        # self.garment_builder = GarmentBuilder(mcfg, garments_dict, garment_smpl_model_dict)
        self.garment_builder = GarmentBuilder(mcfg, body_models_dict, garment_dicts_dir)
        self.body_builder = BodyBuilder(mcfg, body_models_dict, obstacle_dict)

        self.data_path = mcfg.data_root

        self.mcfg = mcfg

    def load_sample(self, fname: str, idx: int, garment_name_full: str, gender: str, betas_id: int) -> HeteroData:
        """
        Build HeteroData object for a single sample
        :param fname: name of the pose sequence relative to self.data_path
        :param idx: index of the frame to load (not used if self.mcfg.wholeseq == True)
        :param garment_name_full: name of the garment to load, can be a comma-separated list of names
        :param betas_id: index of the beta parameters in self.betas_table (only used to generate validation sequences when comparing to snug/ssch)
        :return: HelteroData object (see BodyBuilder.build and GarmentBuilder.build for details)
        """
        sequence = self.sequence_loader.load_sequence(fname, betas_id=betas_id)

        idx = idx // sequence['subsample']

        sample = HeteroData()
        # sample = self.body_builder.build(sample, sequence, idx, gender)

        garment_names = garment_name_full.split(',')
        garment_names = [x.strip() for x in garment_names]

        for garment_name in garment_names:
            sample = self.garment_builder.build(sample, sequence, idx, garment_name)

        if random() > self.mcfg.nobody_freq:
            sample = self.body_builder.build(sample, sequence, idx, gender)

            
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
