import importlib
import os
import pickle
from dataclasses import dataclass, MISSING
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import smplx
import torch
from smplx import SMPL
from torch_geometric.data import HeteroData

from utils.coarse import make_coarse_edges
from utils.common import NodeType, triangles_to_edges, separate_arms
from utils.defaults import DEFAULTS
from utils.io import pickle_load
from utils.mesh_creation import GarmentCreator, obj2template
from utils.datasets import make_obstacle_dict
import warnings

from datasets.postcvpr import VertexBuilder

@dataclass
class Config:
    pose_sequence_path: str = MISSING  #  Path to the pose sequence relative to $DEFAULTS.data_root. Can be either sequence of SMLP parameters of a sequence of meshes depending on the value of pose_sequence_type
    garment_template_path: str = MISSING  # Path to the garment template relative to $DEFAULTS.data_root. Can  be either .obj file or preprocessed or .pkl file (see utils/mesh_creation::obj2template)
    pose_sequence_type: str = "body_model"  # "body_model" | "mesh" if "body_model" the pose_sequence_path is a sequence of SMPL parameters, if "mesh" the pose_sequence_path is a sequence of meshes

    sequence_loader: str = 'cmu_npz_smpl'  # Name of the sequence loader to use 
    body_model_root: str = 'body_models'  # Path to the directory containg body model files, should contain `smpl` and/or `smplx` sub-directories. Relative to $DEFAULTS.data_root/aux_data/
    model_type: str = 'smpl'  # Type of the body model ('smpl' or 'smplx')
    gender: str = 'female' # Gender of the body model ('male' | 'female' | 'neutral')    

    obstacle_dict_file: Optional[str] = None  # Path to the file with auxiliary data for obstacles relative to $DEFAULTS.data_root/aux_data/
    n_coarse_levels: int = 4  # Number of coarse levels with long-range edges
    separate_arms: bool = False  # Whether to separate the arms from the rest of the body (to avoid body self-intersections)
    pinned_verts: bool = True  # Whether to use pinned vertices

    wholeseq: bool = True  
    fps: int = 30 

def create_loader(mcfg: Config):

    garment_template_path = os.path.join(DEFAULTS.data_root, mcfg.garment_template_path)

    if garment_template_path.endswith('.obj'):

        garment_dict = obj2template(garment_template_path)
        warnings.warn("""Loading from garment geometry from .obj. \n
        It may take a while to build coarse edges. \n
        Consider converting the garment to .pkl using utils/mesh_creation::obj2template())""")

    elif garment_template_path.endswith('.pkl'):
        garment_dict = pickle_load(garment_template_path)
    else:
        raise ValueError(f'Unknown garment template format: {mcfg.garment_template_path}, has to be .obj or .pkl')

    if mcfg.pose_sequence_type == 'mesh':
        body_model = None
    elif mcfg.pose_sequence_type == 'body_model':
        body_model_root = os.path.join(DEFAULTS.aux_data, mcfg.body_model_root)
        body_model = smplx.create(body_model_root, model_type=mcfg.model_type, gender=mcfg.gender, use_pca=False)
    else:
        raise ValueError(f'Unknown pose sequence type: {mcfg.pose_sequence_type}, has to be "mesh" or "body_model"')

    obstacle_dict = make_obstacle_dict(mcfg)
    loader = Loader(mcfg, garment_dict, obstacle_dict, body_model)
    return loader


def create(mcfg: Config):
    loader = create_loader(mcfg)

    pose_sequence_path = os.path.join(DEFAULTS.data_root, mcfg.pose_sequence_path)
    dataset = Dataset(loader, pose_sequence_path)
    return dataset



class GarmentBuilder:
    """
    Class to build the garment meshes from SMPL parameters.
    """

    def __init__(self, mcfg: Config, garment_dict: dict):
        """
        :param mcfg: config
        :param garments_dict: dictionary with data for all garments
        """
        self.mcfg = mcfg
        self.garment_dict = garment_dict
        self.vertex_builder = VertexBuilder(mcfg)

        self.gc = GarmentCreator(None, None, None, None, collect_lbs=False, coarse=True, verbose=False)    

    def add_verts(self, sample: HeteroData, garment_dict: dict) -> HeteroData:

        n_frames = sample['obstacle'].lookup.shape[1] + 2

        if 'vertices' in garment_dict:
            pos = garment_dict['vertices']
        else:
            pos = garment_dict['rest_pos']

        pos = torch.FloatTensor(pos)[None,].permute(1, 0, 2)
        pos = pos.repeat(1, n_frames, 1)

        sample['cloth'].prev_pos = pos[:, 0]
        sample['cloth'].pos = pos[:, 1]
        sample['cloth'].target_pos = pos[:, 2]
        sample['cloth'].lookup = pos[:, 2:]
        sample['cloth'].rest_pos = pos[:, 0]

        return sample

    
    def add_vertex_type(self, sample: HeteroData, garment_dict: str) -> HeteroData:
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

        if self.mcfg.pinned_verts and 'node_type' in garment_dict:
            vertex_type = garment_dict['node_type'].astype(np.int64)
        else:
            V = sample['cloth'].pos.shape[0]
            vertex_type = np.zeros((V, 1)).astype(np.int64)

        sample['cloth'].vertex_type = torch.tensor(vertex_type)
        return sample


    def add_faces_and_edges(self, sample: HeteroData, garment_dict: dict) -> HeteroData:
        """
        Add garment faces to `sample['cloth']`
        Add bi-directional edges to `sample['cloth', 'mesh_edge', 'cloth']`

        :param sample: HeteroData
        :param garment_name: name of the garment in `self.garment_smpl_model_dict` and `self.garments_dict`
        :return:
            sample['cloth'].faces_batch: torch.LongTensor [3xF]
            ample['cloth', 'mesh_edge', 'cloth'].edge_index: torch.LongTensor [2xE]
        """

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

    def add_coarse(self, sample: HeteroData, garment_dict: dict) -> HeteroData:
        """
        Add coarse edges to `sample` as `sample['cloth', f'coarse_edge{i}', 'cloth'].edge_index`.
        where `i` is the number of the coarse level (starting from `0`)

        :param sample: HeteroData
        :param garment_name:
        :return: sample['cloth', f'coarse_edge{i}', 'cloth'].edge_index: torch.LongTensor [2, E_i]
        """
        if self.mcfg.n_coarse_levels == 0:
            return sample

        faces = garment_dict['faces']

        # Randomly choose center of the mesh
        # center of a graph is a node with minimal eccentricity (distance to the farthest node)
        if 'center' not in garment_dict:
            garment_dict = self.gc.add_coarse_edges(garment_dict, self.mcfg.n_coarse_levels)

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
    
    def find_closest_faces(self, pinned_pos, obstacle_pos, obstacle_faces, distance_threshold=3e-2):
        N = pinned_pos.size(0)
        F = obstacle_faces.size(0)
        
        # Get the vertices of each face
        v0 = obstacle_pos[obstacle_faces[:, 0]]  # (F, 3)
        v1 = obstacle_pos[obstacle_faces[:, 1]]  # (F, 3)
        v2 = obstacle_pos[obstacle_faces[:, 2]]  # (F, 3)

        face_centers = (v0 + v1 + v2) / 3  # (F, 3)
        
        # Calculate the normal vectors for each face
        v0v1 = v1 - v0  # (F, 3)
        v0v2 = v2 - v0  # (F, 3)
        normals = torch.cross(v0v1, v0v2)  # (F, 3)
        normal_lengths = torch.norm(normals, dim=1, keepdim=True)  # (F, 1)
        
        # Normalize the normals
        normals = normals / normal_lengths  # (F, 3)
        
        # Calculate the distances from each point to each face
        pinned_pos_expanded = pinned_pos.unsqueeze(1).expand(N, F, 3)  # (N, F, 3)
        v0_expanded = v0.unsqueeze(0).expand(N, F, 3)  # (N, F, 3)
        
        vectors_to_points = pinned_pos_expanded - v0_expanded  # (N, F, 3)
        distances_to_planes = torch.abs(torch.sum(vectors_to_points * normals.unsqueeze(0), dim=2)) / normal_lengths.t()  # (N, F)
        distances_to_centers = torch.norm(pinned_pos_expanded - face_centers.unsqueeze(0), dim=2)  # (N, F)

        # Project points onto the plane of each face
        projections = pinned_pos_expanded - normals.unsqueeze(0) * torch.sum(vectors_to_points * normals.unsqueeze(0), dim=2, keepdim=True)  # (N, F, 3)
        
        # Compute barycentric coordinates
        v2v0 = -v0v2  # (F, 3)
        d00 = torch.sum(v0v1 * v0v1, dim=1)  # (F,)
        d01 = torch.sum(v0v1 * v0v2, dim=1)  # (F,)
        d11 = torch.sum(v0v2 * v0v2, dim=1)  # (F,)
        denom = d00 * d11 - d01 * d01  # (F,)
        
        p_to_v0 = projections - v0_expanded  # (N, F, 3)
        d20 = torch.sum(p_to_v0 * v0v1.unsqueeze(0), dim=2)  # (N, F)
        d21 = torch.sum(p_to_v0 * v0v2.unsqueeze(0), dim=2)  # (N, F)
        
        v = (d11 * d20 - d01 * d21) / denom.unsqueeze(0)  # (N, F)
        w = (d00 * d21 - d01 * d20) / denom.unsqueeze(0)  # (N, F)
        u = 1.0 - v - w  # (N, F)
        
        # Check if the projection falls inside the face
        inside_face = (u >= 0) & (v >= 0) & (w >= 0)  # (N, F)
        
        # Set distances to infinity where the projection is outside the face
        distances_to_planes_filtered = torch.where(inside_face, distances_to_planes, torch.full_like(distances_to_planes, float('inf')))

        # Set distances to the centers where the projection is outside all the faces
        # all_noinside_mask = ~inside_face.any(dim=1)
        over_the_threshold_mask = distances_to_centers > distance_threshold
        exclude_pairs_mask = ~inside_face | over_the_threshold_mask
        exclude_nodes_mask = exclude_pairs_mask.all(dim=1)

        distances_to_planes_filtered[exclude_nodes_mask] = distances_to_centers[exclude_nodes_mask]

        # Find the closest face for each point
        closestface_id = torch.argmin(distances_to_planes_filtered, dim=1)  # (N,)

        distances = torch.min(distances_to_planes_filtered, dim=1).values  # (N,)
        
        return closestface_id

    
    def compute_barycentric_coordinates(self, pinned_pos, obstacle_pos, obstacle_faces, closestface_id):
        # Extract the vertices of the closest faces
        v0 = obstacle_pos[obstacle_faces[closestface_id, 0]]  # (N, 3)
        v1 = obstacle_pos[obstacle_faces[closestface_id, 1]]  # (N, 3)
        v2 = obstacle_pos[obstacle_faces[closestface_id, 2]]  # (N, 3)

        # Compute the vectors relative to the vertices of the triangle
        v0v1 = v1 - v0  # (N, 3)
        v0v2 = v2 - v0  # (N, 3)
        v0p = pinned_pos - v0  # (N, 3)

        # Compute dot products
        d00 = torch.sum(v0v1 * v0v1, dim=1)  # (N,)
        d01 = torch.sum(v0v1 * v0v2, dim=1)  # (N,)
        d11 = torch.sum(v0v2 * v0v2, dim=1)  # (N,)
        d20 = torch.sum(v0p * v0v1, dim=1)   # (N,)
        d21 = torch.sum(v0p * v0v2, dim=1)   # (N,)

        # Compute the denominator of the barycentric coordinates
        denom = d00 * d11 - d01 * d01  # (N,)

        # Compute the barycentric coordinates
        v = (d11 * d20 - d01 * d21) / denom  # (N,)
        w = (d00 * d21 - d01 * d20) / denom  # (N,)
        u = 1.0 - v - w  # (N,)

        # Stack the barycentric coordinates into a tensor
        barycoords = torch.stack([u, v, w], dim=1)  # (N, 3)

        # Compute the normal vectors for each face
        normals = torch.cross(v0v1, v0v2)  # (N, 3)
        normal_lengths = torch.norm(normals, dim=1, keepdim=True)  # (N, 1)
        normals = normals / normal_lengths  # (N, 3)

        # Compute the distances from each point to the plane of the face
        ndists = torch.sum((pinned_pos - v0) * normals, dim=1, keepdim=True)  # (N, 1)

        return barycoords, ndists
    

    def compute_pinned_target_pos(self, obstacle_pos_sequence, obstacle_faces, closestface_id, barycoords, ndists):
        obstacle_pos_sequence = obstacle_pos_sequence.permute(1,0,2)

        # Extract the barycentric coordinates
        u = barycoords[:, 0].unsqueeze(0)  # (1, N)
        v = barycoords[:, 1].unsqueeze(0)  # (1, N)
        w = barycoords[:, 2].unsqueeze(0)  # (1, N)

        # Gather the vertices for the closest faces for all frames
        v0_indices = obstacle_faces[closestface_id, 0]  # (N,)
        v1_indices = obstacle_faces[closestface_id, 1]  # (N,)
        v2_indices = obstacle_faces[closestface_id, 2]  # (N,)
        
        v0 = obstacle_pos_sequence[:, v0_indices]  # (K, N, 3)
        v1 = obstacle_pos_sequence[:, v1_indices]  # (K, N, 3)
        v2 = obstacle_pos_sequence[:, v2_indices]  # (K, N, 3)
        
        # Compute the weighted sum of the vertices using the barycentric coordinates
        pinned_pos_bary = u.unsqueeze(2) * v0 + v.unsqueeze(2) * v1 + w.unsqueeze(2) * v2  # (K, N, 3)
        
        # Compute the normal vectors for each face for all frames
        v0v1 = v1 - v0  # (K, N, 3)
        v0v2 = v2 - v0  # (K, N, 3)
        normals = torch.cross(v0v1, v0v2, dim=2)  # (K, N, 3)
        normal_lengths = torch.norm(normals, dim=2, keepdim=True)  # (K, N, 1)
        normals = normals / normal_lengths  # (K, N, 3)
        
        # Compute the normal distances to add to the barycentric coordinates projection
        ndists_expanded = ndists.unsqueeze(0)  # (1, N, 1)


        pinned_target_pos = pinned_pos_bary + normals * ndists_expanded  # (K, N, 3)

        pinned_target_pos = pinned_target_pos.permute(1, 0, 2)
        
        return pinned_target_pos
        
    def update_pinned_target(self, sample: HeteroData) -> HeteroData:
        cloth_target_pos = sample['cloth'].target_pos
        cloth_lookup = sample['cloth'].lookup


        cloth_vertex_type = sample['cloth'].vertex_type
        pinned_mask = cloth_vertex_type == NodeType.HANDLE

        if not pinned_mask.any():
            return sample
        
        pinned_mask = pinned_mask[:, 0]

        


        cloth_first_frame_pos = sample['cloth'].pos
        obstacle_first_frame_pos = sample['obstacle'].pos

        obstacle_faces = sample['obstacle'].faces_batch.T

        pinned_nodes_pos = cloth_first_frame_pos[pinned_mask]

        closest_face_ids = self.find_closest_faces(pinned_nodes_pos, obstacle_first_frame_pos, obstacle_faces)
        barycoords, ndists = self.compute_barycentric_coordinates(pinned_nodes_pos, 
                                                                  obstacle_first_frame_pos, 
                                                                  obstacle_faces, 
                                                                  closest_face_ids)
        

        obstacle_target_pos = sample['obstacle'].target_pos
        obstacle_lookup = sample['obstacle'].lookup
        obstacle_target_plus_lookup = torch.cat([obstacle_target_pos[:, None], obstacle_lookup], dim=1)

        pinned_target_plus_lookup = self.compute_pinned_target_pos(obstacle_target_plus_lookup,
                                                              obstacle_faces,
                                                                closest_face_ids,
                                                                barycoords,
                                                                ndists)
        
        pinned_target_pos = pinned_target_plus_lookup[:, 0]
        pinned_lookup = pinned_target_plus_lookup[:, 1:]
        
        cloth_target_pos[pinned_mask] = pinned_target_pos
        cloth_lookup[pinned_mask] = pinned_lookup

        sample['cloth'].target_pos = cloth_target_pos
        return sample
        
        # Compute the target positions for the pinned nodes


    def build(self, sample: HeteroData) -> HeteroData:
        """
        Add all data for the garment to the sample

        :param sample: HeteroData
        :return:
            sample['cloth'].pos torch.FloatTensor [VxNx3]: vertex positions at the current frame
            sample['cloth'].rest_pos torch.FloatTensor [Vx3]: vertex positions in the canonical pose
            sample['cloth'].faces_batch torch.LongTensor [3xF]: garment faces
            sample['cloth'].vertex_type torch.LongTensor [Vx1]: vertex type (0 - regular, 3 - pinned)
            sample['cloth'].vertex_level torch.LongTensor [Vx1]: level of the vertex in the hierarchy (always 0 for the body)

            sample['cloth', 'mesh_edge', 'cloth'].edge_index: torch.LongTensor [2xE]: mesh edges

            for each coarse level `i` in [0, self.mcfg.n_coarse_levels]:
                sample['cloth', f'coarse_edge{i}', 'cloth'].edge_index: torch.LongTensor [2, E_i]: coarse edges at level `i`

        """
        sample = self.add_verts(sample, self.garment_dict)
        sample = self.add_coarse(sample, self.garment_dict)
        sample = self.add_vertex_type(sample, self.garment_dict)
        sample = self.update_pinned_target(sample)
        sample = self.add_faces_and_edges(sample, self.garment_dict)

        return sample


class SMPLBodyBuilder:
    """
    Class for building body meshed from SMPL parameters
    """

    def __init__(self, mcfg: Config, smpl_model: SMPL, obstacle_dict: dict):
        """
        :param mcfg: Config
        :param smpl_model:
        :param obstacle_dict: auxiliary data for the obstacle
                obstacle_dict['vertex_type']: vertex type (1 - regular obstacle node, 2 - hand node (omitted during inference to avoid body self-penetrations))
        """
        self.smpl_model = smpl_model
        self.obstacle_dict = obstacle_dict
        self.mcfg = mcfg
        self.vertex_builder = VertexBuilder(mcfg)

    
    def make_smpl_vertices(self, sequence_dict, **kwargs) -> np.ndarray:
        """
        Create body vertices from SMPL parameters (used in VertexBuilder.add_verts)

        :param sequence_dict: dict with SMPL parameters:
            body_pose: SMPL pose parameters [Nx69]
            global_orient: SMPL global_orient [Nx3]
            transl: SMPL translation [Nx3]
            betas: SMPL betas [Nx10]

        :return: vertices [NxVx3]
        """

        input_dict = {k: torch.FloatTensor(v) for k, v in sequence_dict.items()}
        with torch.no_grad():
            smpl_output = self.smpl_model(**input_dict)
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
        faces = torch.LongTensor(self.smpl_model.faces.astype(np.int64))
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

        :return:
            sample['obstacle'].prev_pos torch.FloatTensor [VxNx3]: vertex positions at the previous frame
            sample['obstacle'].pos torch.FloatTensor [VxNx3]: vertex positions at the current frame
            sample['obstacle'].target_pos torch.FloatTensor [VxNx3]: vertex positions at the next frame

            sample['obstacle'].faces_batch torch.LongTensor [3xF]: garment faces
            sample['obstacle'].vertex_type torch.LongTensor [Vx1]: vertex type (1 - regular obstacle, 2 - omitted)
            sample['obstacle'].vertex_level torch.LongTensor [Vx1]: level of the vertex in the hierarchy (always 0 for the body)

        """
        sample = self.vertex_builder.add_verts(sample, sequence_dict, 0,  self.make_smpl_vertices, "obstacle")
        sample = self.add_vertex_type(sample)
        sample = self.add_faces(sample)
        sample = self.add_vertex_level(sample)
        return sample


class BareMeshBodyBuilder:
    """
    Class for building body meshed from SMPL parameters
    """

    def __init__(self, mcfg: Config, obstacle_dict: dict):
        """
        :param mcfg: Config
        :param obstacle_dict: auxiliary data for the obstacle
                obstacle_dict['vertex_type']: vertex type (1 - regular obstacle node, 2 - hand node (omitted during inference to avoid body self-penetrations))
        """
        self.obstacle_dict = obstacle_dict
        self.mcfg = mcfg
        self.vertex_builder = VertexBuilder(mcfg)

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

    def add_faces(self, sample: HeteroData, sequence_dict: dict) -> HeteroData:
        """
        Add body faces to the obstacle object in the sample
        """
        faces = torch.LongTensor(sequence_dict['faces'].astype(np.int64))
        sample['obstacle'].faces_batch = faces.T
        return sample

    def add_verts(self, sample: HeteroData, sequence_dict: dict) -> HeteroData:
        """
        Add body vertices to the obstacle object in the sample
        """

        pos = torch.FloatTensor(sequence_dict["verts"]).permute(1, 0, 2)

        sample['obstacle'].prev_pos = pos[:, 0]
        sample['obstacle'].pos = pos[:, 1]
        sample['obstacle'].target_pos = pos[:, 2]
        sample['obstacle'].lookup = pos[:, 2:]

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

        :return:
            sample['obstacle'].prev_pos torch.FloatTensor [VxNx3]: vertex positions at the previous frame
            sample['obstacle'].pos torch.FloatTensor [VxNx3]: vertex positions at the current frame
            sample['obstacle'].target_pos torch.FloatTensor [VxNx3]: vertex positions at the next frame

            sample['obstacle'].faces_batch torch.LongTensor [3xF]: garment faces
            sample['obstacle'].vertex_type torch.LongTensor [Vx1]: vertex type (1 - regular obstacle, 2 - omitted)
            sample['obstacle'].vertex_level torch.LongTensor [Vx1]: level of the vertex in the hierarchy (always 0 for the body)

        """
        sample = self.add_verts(sample, sequence_dict)
        sample = self.add_vertex_type(sample)
        sample = self.add_faces(sample, sequence_dict)
        sample = self.add_vertex_level(sample)
        return sample

class MeshSequenceLoader:
    def __init__(self, mcfg):
        self.mcfg = mcfg

    def load_sequence(self, fname: str) -> dict:

        with open(fname, 'rb') as f:
            sequence = pickle.load(f)

        return sequence


class Loader:
    """
    Class for building HeteroData objects containing all data for a single sample
    """

    def __init__(self, mcfg: Config, garment_dict: dict, obstacle_dict: dict, smpl_model: SMPL = None):

        self.garment_builder = GarmentBuilder(mcfg, garment_dict)

        if mcfg.pose_sequence_type == 'body_model':
            self.body_builder = SMPLBodyBuilder(mcfg, smpl_model, obstacle_dict)
            sequence_loader_module = importlib.import_module(f'datasets.sequence_loaders.{mcfg.sequence_loader}')
            SequenceLoader = sequence_loader_module.SequenceLoader
            self.sequence_loader = SequenceLoader(mcfg, '')
        elif mcfg.pose_sequence_type == 'mesh':
            self.body_builder = BareMeshBodyBuilder(mcfg, obstacle_dict)
            self.sequence_loader = MeshSequenceLoader(mcfg)
        else:
            raise ValueError(f'Unknown pose sequence type {mcfg.pose_sequence_type}. Should be "body_model" or "mesh"')


    def load_sample(self, fname: str) -> HeteroData:
        """
        Build HeteroData object for a single sample
        :param fname: path to the pose sequence file
        :return: HelteroData object (see BodyBuilder.build and GarmentBuilder.build for details)
        """
        sequence = self.sequence_loader.load_sequence(fname)
        sample = HeteroData()
        sample = self.body_builder.build(sample, sequence)
        sample = self.garment_builder.build(sample)
        return sample


class Dataset:
    def __init__(self, loader: Loader, pose_sequence_path: str):
        """
        Dataset class for building training and validation samples
        :param loader: Loader object
        """

        self.loader = loader
        self.pose_sequence_path = pose_sequence_path

        self._len = 1


    def __getitem__(self, item: int) -> HeteroData:
        """
        Load a sample given a global index
        """

        sample = self.loader.load_sample(self.pose_sequence_path)
        sample['sequence_name'] = self.pose_sequence_path
        sample['garment_name'] = 'stub'

        return sample

    def __len__(self) -> int:
        return self._len
