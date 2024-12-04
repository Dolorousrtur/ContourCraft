import glob
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass, MISSING
from typing import Optional, Dict

import numpy as np
import smplx
import torch
from smplx import SMPLX
from torch_geometric.data import HeteroData

from utils.io import load_obj, pickle_load
from utils.coarse import make_coarse_edges
from utils.common import triangles_to_edges, NodeType
from utils.defaults import DEFAULTS
from utils.mesh_creation import add_coarse_edges
import torch_geometric

@dataclass
class Config:
    garment_dict_path: str = MISSING  # Path to the garment dictionary
    garment_name_list: str = MISSING  # Name of the garments in the garment dictionary, separated by commas
    smplx_model: str = MISSING  # Path to the SMPL-x model
    pose_path: Optional[str] = None  # Path to the SMPL pose file (only used for betas)
    garment_init_obj_path_list: Optional[str] = None  # Path to the garment obj files, separated by commas

    
    n_frames: int = 30  # Number of frames in the sequence
    obstacle_dict_file: Optional[
        str] = None  # Path to the file with auxiliary data for obstacles relative to $HOOD_DATA/aux_data/
    pinned_verts: bool = True  # Whether to use pinned vertices
    restpos_scale: float = 1.  # Minimum scale for randomly sampling the canonical garment geometry
    n_coarse_levels: int = 1  # Number of coarse levels with long-range edges


def make_obstacle_dict(mcfg: Config) -> dict:
    if mcfg.obstacle_dict_file is None:
        return {}

    obstacle_dict_path = os.path.join(DEFAULTS.aux_data, mcfg.obstacle_dict_file)
    with open(obstacle_dict_path, 'rb') as f:
        obstacle_dict = pickle.load(f)
    return obstacle_dict


def create_loader(mcfg: Config):
    smplx_model_path = mcfg.smplx_model
    # smplx_model = smplx.SMPLX(smplx_model_path, use_pca=False)
    smplx_model = smplx.SMPLX(smplx_model_path, use_pca=True, num_pca_comps=12)

    obstacle_dict = make_obstacle_dict(mcfg)
    garment_dict = pickle_load(mcfg.garment_dict_path)

    print('GARMENT DICT', mcfg.garment_dict_path)
    print('00176_Template', garment_dict['00176_Template'].keys())

    garment_name_list = mcfg.garment_name_list.split(',')

    garment_init_verts_list = []


    if mcfg.garment_init_obj_path_list is None:
        for gname in garment_name_list:
            garment_dict[gname]['init_pos'] = garment_dict[gname]['lbs']['v']
    else:
        garment_init_obj_path_list = mcfg.garment_init_obj_path_list.split(',')
        for garment_init_obj_path in garment_init_obj_path_list:
            garment_init_verts, _ = load_obj(garment_init_obj_path)
            garment_init_verts_list.append(garment_init_verts)

        for gname, ginit in zip(garment_name_list, garment_init_verts_list):
            garment_dict[gname]['init_pos'] = ginit

    loader = Loader(mcfg, smplx_model, garment_name_list, obstacle_dict=obstacle_dict, garment_dict=garment_dict)
    return loader


def create(mcfg: Config):
    loader = create_loader(mcfg)
    garment_name = mcfg.garment_name_list
    dataset = Dataset(loader, garment_name)
    return dataset


class VertexBuilder:
    """
    Helper class to build garment and body vertices from a sequence of SMPL poses.
    """

    def __init__(self, mcfg):
        self.mcfg = mcfg

    # @staticmethod
    def build(self, sequence_dict: dict, f_make, idx_start: int, idx_end: int = None) -> np.ndarray:
        """
        Build vertices from a sequence of SMPL poses using the given `f_make` function.
        :param sequence_dict: a dictionary of SMPL parameters
        :param f_make: a function that takes SMPL parameters and returns vertices
        :param idx_start: first frame index
        :param idx_end: last frame index
        :return: [Nx3] mesh vertices
        """

        if 'betas' in sequence_dict:
            betas = sequence_dict['betas']
            if len(betas.shape) == 2 and betas.shape[0] != 1:
                betas = betas[idx_start: idx_end]

            input_dict = {}
            input_dict['betas'] = betas

            for k in ['global_orient',
                    'body_pose',
                    'transl',
                    'left_hand_pose',
                    'right_hand_pose',
                    'jaw_pose',
                    'leye_pose',
                    'reye_pose',
                    'expression']:
                input_dict[k] = sequence_dict[k][idx_start: idx_end]

            verts = f_make(input_dict)
        else:
            verts = f_make({})
            verts = np.concatenate([verts]*self.mcfg.n_frames, axis=0)

        return verts

    def pos2tensor(self, pos: np.ndarray) -> torch.Tensor:
        """
        Convert a numpy array of vertices to a tensor and permute the axes into [VxNx3] (torch geometric format)
        """
        pos = torch.tensor(pos).permute(1, 0, 2)
        return pos

    def add_verts(self, sample: HeteroData, sequence_dict: dict, f_make, object_key: str,
                  **kwargs) -> HeteroData:
        """
        Builds the vertices from the given SMPL pose sequence and adds them to the HeteroData sample.
        :param sample: HetereoData object
        :param sequence_dict: sequence of SMPL parameters
        :param f_make: function that takes SMPL parameters and returns vertices
        :param object_key: name of the object to build vertices for ('cloth' or 'obstacle')
        :return: updated HeteroData object
        """

        pos_dict = {}

        # Build the vertices for the whole sequence

        # if 'betas' not in sequence_dict:
        #     assert False
        all_vertices = self.build(sequence_dict, f_make, 0, None,
                                           **kwargs)
        pos_dict['prev_pos'] = all_vertices
        pos_dict['pos'] = all_vertices
        pos_dict['target_pos'] = all_vertices

        for k, v in pos_dict.items():
            v = self.pos2tensor(v)
            setattr(sample[object_key], k, v)

        return sample


class GarmentBuilder:
    """
    Class to build the garment meshes from SMPL parameters.
    """

    def __init__(self, mcfg: Config, garment_dict: dict):
        """
        :param mcfg: config
        :param garments_dict: dictionary with data for all garments
        :param garment_smpl_model_dict: dictionary with SMPL models for all garments
        """
        self.mcfg = mcfg
        self.garment_dict = garment_dict

        self.vertex_builder = VertexBuilder(mcfg)

    def add_verts(self, sample: HeteroData, garment_name) -> HeteroData:
        garment_dict = self.garment_dict[garment_name]
        rest_pos = garment_dict['rest_pos']
        rest_pos = torch.FloatTensor(rest_pos)

        pos = torch.FloatTensor(garment_dict['init_pos'])
        pos = pos.unsqueeze(1).repeat(1, self.mcfg.n_frames, 1)

        sample['cloth'].prev_pos = pos
        sample['cloth'].pos = pos
        sample['cloth'].target_pos = pos
        sample['cloth'].rest_pos = rest_pos

        return sample

    def add_vertex_type(self, sample: HeteroData, garment_name) -> HeteroData:
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
        garment_dict = self.garment_dict[garment_name]

        V = sample['cloth'].pos.shape[0]
        vertex_type = np.zeros((V, 1)).astype(np.int64)


        print('self.mcfg.pinned_verts', self.mcfg.pinned_verts)
        print('garment_dict', garment_dict.keys())

        if self.mcfg.pinned_verts and 'pinned_ids' in garment_dict:
            vertex_type[garment_dict['pinned_ids']] = NodeType.HANDLE

        print('vertex_type', np.unique(vertex_type))

        sample['cloth'].vertex_type = torch.tensor(vertex_type)
        return sample

    def resize_restpos(self, restpos: np.array) -> np.array:
        """
        Randomly resize resting geometry of a garment
        with scale from `self.mcfg.restpos_scale_min` to `self.mcfg.restpos_scale_max`

        :param restpos: Vx3
        :return: resized restpos: Vx3
        """
        mean = restpos.mean(axis=0, keepdims=True)
        restpos -= mean
        restpos *= self.mcfg.restpos_scale
        restpos += mean

        return restpos

    def add_faces_and_edges(self, sample: HeteroData, garment_name) -> HeteroData:
        """
        Add garment faces to `sample['cloth']`
        Add bi-directional edges to `sample['cloth', 'mesh_edge', 'cloth']`

        :param sample: HeteroData
        :param garment_name: name of the garment in `self.garment_smpl_model_dict` and `self.garments_dict`
        :return:
            sample['cloth'].faces_batch: torch.LongTensor [3xF]
            ample['cloth', 'mesh_edge', 'cloth'].edge_index: torch.LongTensor [2xE]
        """
        garment_dict = self.garment_dict[garment_name]

        faces = torch.LongTensor(garment_dict['faces'])
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

    def add_coarse(self, sample: HeteroData, garment_name) -> HeteroData:
        """
        Add coarse edges to `sample` as `sample['cloth', f'coarse_edge{i}', 'cloth'].edge_index`.
        where `i` is the number of the coarse level (starting from `0`)

        :param sample: HeteroData
        :param garment_name:
        :return: sample['cloth', f'coarse_edge{i}', 'cloth'].edge_index: torch.LongTensor [2, E_i]
        """
        garment_dict = self.garment_dict[garment_name]
        if self.mcfg.n_coarse_levels == 0:
            return sample

        faces = garment_dict['faces']

        # Randomly choose center of the mesh
        # center of a graph is a node with minimal eccentricity (distance to the farthest node)
        if 'center' not in garment_dict:
            garment_dict = add_coarse_edges(garment_dict, self.mcfg.n_coarse_levels)

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


    def build(self, sample: HeteroData, garment_name) -> HeteroData:
        """
        Add all data for the garment to the sample

        :param sample: HeteroData
        :return:
            sample['cloth'].pos torch.FloatTensor [VxNx3]: vertex positions at the current frame
            sample['cloth'].rest_pos torch.FloatTensor [Vx3]: vertex positions in the canonical pose
            sample['cloth'].faces_batch torch.LongTensor [3xF]: garment faces
            sample['cloth'].vertex_type torch.LongTensor [Vx1]: vertex type (0 - regular, 3 - pinned)
            sample['obstacle'].vertex_level torch.LongTensor [Vx1]: level of the vertex in the hierarchy (always 0 for the body)

            sample['cloth', 'mesh_edge', 'cloth'].edge_index: torch.LongTensor [2xE]: mesh edges

            for each coarse level `i` in [0, self.mcfg.n_coarse_levels]:
                sample['cloth', f'coarse_edge{i}', 'cloth'].edge_index: torch.LongTensor [2, E_i]: coarse edges at level `i`

        """

        sample_temp = HeteroData()
        sample_temp = self.add_verts(sample_temp, garment_name)
        sample_temp = self.add_coarse(sample_temp, garment_name)
        sample_temp = self.add_vertex_type(sample_temp, garment_name)
        sample_temp = self.add_faces_and_edges(sample_temp, garment_name)

        sample = self.add_garment_to_sample(sample, sample_temp)

        return sample


class BodyBuilder:
    """
    Class for building body meshed from SMPL parameters
    """

    def __init__(self, mcfg: Config, smplx_model: SMPLX, obstacle_dict: dict):
        """
        :param mcfg: Config
        :param smplx_model:
        :param obstacle_dict: auxiliary data for the obstacle
                obstacle_dict['vertex_type']: vertex type (1 - regular obstacle node, 2 - hand node (omitted during inference to avoid body self-penetrations))
        """
        self.smplx_model = smplx_model
        self.obstacle_dict = obstacle_dict
        self.mcfg = mcfg
        self.vertex_builder = VertexBuilder(mcfg)

    def make_smpl_vertices(self, sequence_dict, **kwargs) -> np.ndarray:
        """
        Create body vertices from SMPL parameters (used in VertexBuilder.add_verts)

        :param body_pose: SMPL pose parameters [Nx69] OR [69]
        :param global_orient: SMPL global_orient [Nx3] OR [3]
        :param transl: SMPL translation [Nx3] OR [3]
        :param betas: SMPL betas [Nx10] OR [10]

        :return: vertices [NxVx3]
        """

        input_dict = {k: torch.FloatTensor(v) for k, v in sequence_dict.items()}
        with torch.no_grad():
            smpl_output = self.smplx_model(**input_dict)
        vertices = smpl_output.vertices.numpy().astype(np.float32)

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

    def add_faces(self, sample: HeteroData) -> HeteroData:
        """
        Add body faces to the obstacle object in the sample
        """
        faces = torch.LongTensor(self.smplx_model.faces.astype(np.int64))
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

    def build(self, sample: HeteroData, sequence_dict: dict) -> HeteroData:
        """
        Add all data for the body (obstacle) to the sample
        :param sample: HeteroData object to add data to
        :param sequence_dict: dict with SMPL parameters
        :param idx: index of the current frame in the sequence
        
        :return:
            sample['obstacle'].prev_pos torch.FloatTensor [VxNx3]: vertex positions at the previous frame
            sample['obstacle'].pos torch.FloatTensor [VxNx3]: vertex positions at the current frame
            sample['obstacle'].target_pos torch.FloatTensor [VxNx3]: vertex positions at the next frame

            sample['obstacle'].faces_batch torch.LongTensor [3xF]: garment faces
            sample['obstacle'].vertex_type torch.LongTensor [Vx1]: vertex type (1 - regular obstacle, 2 - omitted)
            sample['obstacle'].vertex_level torch.LongTensor [Vx1]: level of the vertex in the hierarchy (always 0 for the body)

        """
        sample = self.vertex_builder.add_verts(sample, sequence_dict, self.make_smpl_vertices, "obstacle")
        sample = self.add_vertex_type(sample)
        sample = self.add_faces(sample)
        sample = self.add_vertex_level(sample)
        return sample


class SequenceLoader:
    def __init__(self, mcfg, data_path):
        self.mcfg = mcfg
        self.data_path = data_path

    def build_sequence(self, pose_path) -> dict:
        if pose_path is None:
            return {}

        sequence = defaultdict(list)
        n_frames = self.mcfg.n_frames

      
        with open(pose_path, 'rb') as f:
            data = pickle.load(f)
        for key in data:
            sequence[key] = [data[key]] * n_frames

        for key in sequence:
            sequence[key] = np.stack(sequence[key])

        
        for key in sequence:
            if key != 'betas':
                sequence[key]*=0

        return sequence

    def load_sequence(self) -> dict:
        sequence = self.build_sequence(self.data_path)

        return sequence


class Loader:
    """
    Class for building HeteroData objects containing all data for a single sample
    """

    def __init__(self, mcfg: Config, smplx_model: SMPLX, garment_names, garment_dict: dict, obstacle_dict: dict):
        self.sequence_loader = SequenceLoader(mcfg, mcfg.pose_path)
        self.garment_builder = GarmentBuilder(mcfg, garment_dict)
        self.body_builder = BodyBuilder(mcfg, smplx_model, obstacle_dict)

        self.garment_names = garment_names

        self.data_path = mcfg.pose_path

    def load_sample(self: str) -> HeteroData:
        """
        Build HeteroData object for a single sample
        :param seq_dir: name of the pose sequence relative to self.data_path
        :param garment_name: name of the garment to load
        :return: HelteroData object (see BodyBuilder.build and GarmentBuilder.build for details)
        """
        sequence = self.sequence_loader.load_sequence()


        sample = HeteroData()
        sample = self.body_builder.build(sample, sequence)

        for gname in self.garment_names:
            sample = self.garment_builder.build(sample, gname)
        return sample


class Dataset:
    def __init__(self, loader: Loader, garment_name):
        """
        Dataset class for building training and validation samples
        :param loader: Loader object
        """
        self.loader = loader
        self.garment_name = garment_name
        self._len = 1

    def __getitem__(self, item: int) -> HeteroData:
        """
        Load a sample given a global index
        """


        sample = self.loader.load_sample()
        # sample['sequence_name'] = self.seq_dir
        sample['garment_name'] = self.garment_name
        return sample

    def __len__(self) -> int:
        return self._len
