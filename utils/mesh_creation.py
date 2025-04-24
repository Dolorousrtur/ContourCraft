import itertools
import os
from pathlib import Path
import pickle

import networkx as nx
import numpy as np
import smplx
import trimesh
from sklearn import neighbors
from tqdm import tqdm
import torch

from utils.io import load_obj, pickle_dump, pickle_load, save_obj
from utils.coarse import make_graph_from_faces, make_coarse_edges
from utils.common import NodeType
from utils.defaults import DEFAULTS
from utils.lbs import makeT
from utils import lbs as my_lbs
from smplx.lbs import blend_shapes
from utils.simulator import Simulator


def add_pinned_verts(garment_dict_path, pinned_indices):
    """
    Modify `node_type` field in the pickle file to mark pinned vertices with NodeType.HANDLE
    :param garment_dict_path: path to the garment dict file
    :param pinned_indices: list of pinned vertex indices
    """

    pkl = pickle_load(garment_dict_path)
    node_type = np.zeros_like(pkl['rest_pos'][:, :1])
    node_type[pinned_indices] = NodeType.HANDLE
    node_type = node_type.astype(np.int64)
    pkl['node_type'] = node_type
    pickle_dump(pkl, garment_dict_path)


def add_buttons(file, button_edges):
    with open(file, 'rb') as f:
        pkl = pickle.load(f)

    pkl['button_edges'] = button_edges

    with open(file, 'wb') as f:
        pickle.dump(pkl, f)


def sample_skinningweights(points, smpl_tree, sigmas, smpl_model):
    """
    For each point in the garment samples a random point for a Gaussian distribution around it,
    finds the nearest SMPL vertex and returns the corresponding skinning weights

    :param points: garment vertices
    :param smpl_tree: sklearn.neighbors.KDTree with SMPL vertex positions
    :param sigmas: standard deviation of the Gaussian distributions
    :param smpl_model: SMPL model
    :return: garment_shapedirs: shape blend shapes for the garment vertices
    :return: garment_posedirs: pose blend shapes for the garment vertices
    :return: garment_lbs_weights: skinning weights for the garment vertices
    """
    noise = np.random.randn(*points.shape)
    points_sampled = noise * sigmas + points
    _, nn_list = smpl_tree.query(points_sampled)
    nn_inds = nn_list[..., 0]

    garment_shapedirs = smpl_model.shapedirs[nn_inds].cpu().numpy()

    N = smpl_model.posedirs.shape[0]
    garment_posedirs = smpl_model.posedirs.reshape(N, -1, 3)[:, nn_inds].reshape(N, -1).cpu().numpy()
    garment_lbs_weights = smpl_model.lbs_weights[nn_inds].cpu().numpy()

    return garment_shapedirs, garment_posedirs, garment_lbs_weights


def approximate_graph_center(G):
    """
    Approximates the center of a graph using the two-sweep BFS algorithm.

    Returns:
    center_vertex: The approximate center vertex of the graph.
    """

    # First BFS from an arbitrary vertex
    start_vertex = list(G.nodes())[0]
    lengths = nx.single_source_shortest_path_length(G, start_vertex)
    u = max(lengths, key=lengths.get)

    # Second BFS from vertex u
    lengths = nx.single_source_shortest_path_length(G, u)
    w = max(lengths, key=lengths.get)

    # Get the shortest path from u to w
    path = nx.shortest_path(G, source=u, target=w)

    # Find the middle vertex (or vertices) of the path
    path_length = len(path)
    center_index = path_length // 2
    center_vertex = path[center_index]
    return center_vertex

class GarmentCreator:
    def __init__(self, garment_dicts_dir, body_models_root, model_type, gender, 
                 collect_lbs=True, n_samples_lbs=0, coarse=True, n_coarse_levels=4, 
                 approximate_center=False, verbose=False, add_uv=False):
        
        self.garment_dicts_dir = garment_dicts_dir
        self.body_models_root = body_models_root
        self.model_type = model_type
        self.gender = gender
        self.add_uv = add_uv


        self.collect_lbs = collect_lbs
        self.n_lbs_samples = n_samples_lbs
        self.coarse = coarse
        self.n_coarse_levels = n_coarse_levels
        self.verbose = verbose
        self.approximate_center = approximate_center

        if body_models_root is not None:
            self.body_model = smplx.create(body_models_root, model_type, gender=gender, use_pca=False)



    def add_coarse_edges(self, garment_dict):
        n_levels = self.n_coarse_levels

        faces = garment_dict['faces']
        G = make_graph_from_faces(faces)

        components = list(nx.connected_components(G))

        cGd_list = []
        for component in components:
            cg_dict = dict()


            cG = G.subgraph(component)
            component_ids = np.array(list(component))
            faces_mask = np.isin(faces, component_ids).all(axis=1)

            faces_component = faces[faces_mask]

            if self.approximate_center:
                center_nodes = [approximate_graph_center(cG)]
            else:
                center_nodes = nx.center(cG)

            cg_dict['center'] = center_nodes
            cg_dict['coarse_edges'] = dict()

            for center in center_nodes[:3]:
                coarse_edges_dict = make_coarse_edges(faces_component, center, n_levels=n_levels)
                cg_dict['coarse_edges'][center] = coarse_edges_dict
            cGd_list.append(cg_dict)

        cGdk_list = [d['coarse_edges'].keys() for d in cGd_list]
        ctuples = list(itertools.product(*cGdk_list))


        center_list = []
        coarse_edges_dict = dict()
        for ci, ctuple in enumerate(ctuples):
            center_list.append(ci)
            coarse_edges_dict[ci] = dict()

            for l in range(n_levels):
                ce_list = []
                for i, d in enumerate(cGd_list):
                    ce_list.append(d['coarse_edges'][ctuple[i]][l])

                ce_list = np.concatenate(ce_list, axis=0)

                coarse_edges_dict[ci][l] = ce_list


        garment_dict['center'] = np.array(center_list)
        garment_dict['coarse_edges'] = coarse_edges_dict

        return garment_dict

    def make_lbs_dict(self, garment_template_verts, garment_faces):
        """
        Collect linear blend skinning weights for a garment mesh
        :param obj_file:
        :param smpl_file:
        :param n_samples:
        :return:
        """

        body_verts_rest_pose = self.body_model().vertices[0].detach().cpu().numpy()
        n_samples = self.n_lbs_samples

        body_verts_tree = neighbors.KDTree(body_verts_rest_pose)
        distances, nn_list = body_verts_tree.query(garment_template_verts)

        nn_inds = nn_list[..., 0]

        n_posedirs = self.body_model.posedirs.shape[0]

        if n_samples == 0:
            # Take weights of the closest SMPL vertex
            garment_shapedirs = self.body_model.shapedirs[nn_inds].cpu().numpy()
            garment_posedirs = self.body_model.posedirs.reshape(n_posedirs, -1, 3)[:, nn_inds].reshape(n_posedirs, -1).cpu().numpy()
            garment_lbs_weights = self.body_model.lbs_weights[nn_inds].cpu().numpy()
        else:
            garment_shapedirs = 0
            garment_posedirs = 0
            garment_lbs_weights = 0

            # Randomly sample n_samples from a normal distribution with std = distance to the closest SMPL vertex
            # around the garment node and take the average of the weights for the closest SMPL nodes
            # Following "Self-Supervised Collision Handling via Generative 3D Garment Models for Virtual Try-On" [Santesteban et al. 2021]
            for _ in tqdm(range(n_samples)):
                garment_shapedirs_sampled, garment_posedirs_sampled, garment_lbs_weights_sampled = sample_skinningweights(
                    garment_template_verts, body_verts_tree, distances ** 0.5, self.body_model)
                garment_shapedirs += garment_shapedirs_sampled
                garment_posedirs += garment_posedirs_sampled
                garment_lbs_weights += garment_lbs_weights_sampled

            garment_shapedirs = garment_shapedirs / n_samples
            garment_posedirs = garment_posedirs / n_samples
            garment_lbs_weights = garment_lbs_weights / n_samples

        out_dict = dict(v=garment_template_verts, f=garment_faces, shapedirs=garment_shapedirs,
                        posedirs=garment_posedirs, lbs_weights=garment_lbs_weights)

        return out_dict
    
    def _load_from_obj(self, obj_file):
        obj_dict = {}

        if self.add_uv:
            vertices_full, faces_full, verts_uv, faces_uv = load_obj(obj_file, tex_coords=True)
            obj_dict['verts_uv'] = verts_uv
            obj_dict['faces_uv'] = faces_uv
        else:
            vertices_full, faces_full = load_obj(obj_file, tex_coords=False)

        
        obj_dict['vertices'] = vertices_full
        obj_dict['faces'] = faces_full

        return obj_dict
    
    def _make_garment_dict_from_verts(self, obj_dict, vertices_canonical=None):
        vertices_full = obj_dict['vertices']
        faces_full = obj_dict['faces']

        if vertices_canonical is None:
            vertices_canonical = vertices_full
        garment_dict = make_restpos_dict(vertices_canonical, faces_full)

        if self.collect_lbs:
            if self.verbose:
                print('Sampling LBS weights...')
            lbs = self.make_lbs_dict(vertices_full, faces_full)
            garment_dict['lbs'] = lbs
            if self.verbose:
                print('Done.')

        if self.coarse:
            if self.verbose:
                print('Adding coarse edges... (may take a while)')
            garment_dict = self.add_coarse_edges(garment_dict)
            if self.verbose:
                print('Done.')

        garment_dict['gender'] = self.gender
        garment_dict['model_type'] = self.model_type

        if 'verts_uv' in obj_dict:
            garment_dict['verts_uv'] = obj_dict['verts_uv']
            garment_dict['faces_uv'] = obj_dict['faces_uv']

        return garment_dict

    def make_garment_dict(self, obj_file, rest_pos_objfile=None):
        """
        Create a dictionary for a garment from an obj file
        """

        obj_dict = self._load_from_obj(obj_file)

        vertices_canonical = None
        if rest_pos_objfile is not None:
            rest_pos_obj_dict = self._load_from_obj(rest_pos_objfile)
            vertices_canonical = rest_pos_obj_dict['vertices']

        garment_dict = self._make_garment_dict_from_verts(obj_dict, 
                                                          vertices_canonical=vertices_canonical)

        return garment_dict

    def add_garment(self, objfile, garment_name, rest_pos_objfile=None, pinned_indices=None):
        """
        Add a new garment from a given obj file to the garments_dict_file

        :param objfile: path to the obj file with the new garment
        :param garment_dict_path: path to the garment dict file to be created
        """

        garment_dict_path = Path(self.garment_dicts_dir) / f'{garment_name}.pkl'
        garment_dict = self.make_garment_dict(objfile, rest_pos_objfile=rest_pos_objfile)
        if pinned_indices is not None:
            garment_dict = self._add_pinned_verts_dict(garment_dict, pinned_indices)
        pickle_dump(garment_dict, garment_dict_path)
        print(f'Garment dict saved to {garment_dict_path}')


    def add_posed_garment(self, objfile, garment_name, body_params_file, checkpoint_path, n_relaxation_steps=30, pinned_indices=None):
        garment_dict_path = Path(self.garment_dicts_dir) / f'{garment_name}.pkl'
        obj_dict = self._load_from_obj(objfile)

        body_params_dict = pickle_load(body_params_file)

        vertices_posed = obj_dict['vertices']
        vertices_unposed = self.unpose_garment(vertices_posed, body_params_dict)

        obj_dict['vertices'] = vertices_unposed
        garment_dict = self._make_garment_dict_from_verts(obj_dict, vertices_canonical=vertices_posed)

        if pinned_indices is not None:
            garment_dict = self._add_pinned_verts_dict(garment_dict, pinned_indices)

        pickle_dump(garment_dict, garment_dict_path)

        print("Relaxing the garment to remove unposing artifacts...")
        trajectories_dict = self.relax_zeropos(garment_name, checkpoint_path, n_steps=n_relaxation_steps)
        print(f'Garment dict saved to {garment_dict_path}')

        return trajectories_dict     


    def _add_pinned_verts_dict(self, garment_dict, pinned_indices):
        node_type = np.zeros_like(garment_dict['rest_pos'][:, :1])
        node_type[pinned_indices] = NodeType.HANDLE
        node_type = node_type.astype(np.int64)
        garment_dict['node_type'] = node_type
        return garment_dict


    def add_pinned_verts(self, garment_name, pinned_indices):
        """
        Modify `node_type` field in the pickle file to mark pinned vertices with NodeType.HANDLE
        """

        garment_dict_path = Path(self.garment_dicts_dir) / f'{garment_name}.pkl'
        garment_dict = pickle_load(garment_dict_path)
        garment_dict = self._add_pinned_verts_dict(garment_dict, pinned_indices)
        pickle_dump(garment_dict, garment_dict_path)


    def relax_zeropos(self, garment_name, checkpoint_path, n_steps=30):

        sequence_loader = 'cmu_npz_smplx_zeropos' if self.model_type == 'smplx' else 'cmu_npz_smpl_zeropos'
        simulator = Simulator(checkpoint_path, sequence_loader=sequence_loader)

        trajectories_dict = simulator.run_zeropos(garment_name, n_steps=n_steps, gender=self.gender,
                                                   garment_dicts_dir=self.garment_dicts_dir)
        relaxed_verts = trajectories_dict['pred'][-1]

        garment_dict_path = Path(self.garment_dicts_dir) / f'{garment_name}.pkl'
        garment_dict = pickle_load(garment_dict_path)

        faces = garment_dict['faces']
        garment_dict['lbs'] = self.make_lbs_dict(relaxed_verts, faces)

        pickle_dump(garment_dict, garment_dict_path)

        return trajectories_dict

    def unpose_garment(self, garment_verts, smplx_dict):

        smplx_dict_pt = {k: torch.tensor(v).float().cuda() for k, v in smplx_dict.items()}
        body_model = self.body_model.cuda()

        body_verts = body_model(**smplx_dict_pt).vertices[0]
        body_verts = body_verts.detach().cpu().numpy()



        r_permute = np.array([[1,0,0],
                [0,0,-1],
                [0,1,0]], dtype=body_verts.dtype)
        # body_verts = body_verts @ r_permute
        garment_verts = garment_verts @ r_permute.T

        garment_verts = garment_verts - smplx_dict['transl']
        body_verts = body_verts -  smplx_dict['transl']

        smpl_tree = neighbors.KDTree(body_verts)
        distances, nn_list = smpl_tree.query(garment_verts)

        out_temp_bory = '/data/agrigorev/02_Projects/ccraft_data/examples/unpose/smpl_body_temp.obj'
        save_obj(out_temp_bory, body_verts, body_model.faces)

        n_samples = self.n_lbs_samples
        if n_samples == 0:
            garment_shapedirs, garment_posedirs, garment_lbs_weights = sample_skinningweights(
                    garment_verts, smpl_tree, distances * 0, body_model)

            garment_shapedirs = torch.tensor(garment_shapedirs)
            garment_posedirs = torch.tensor(garment_posedirs)
            garment_lbs_weights = torch.tensor(garment_lbs_weights)
        else:
            garment_shapedirs = 0
            garment_posedirs = 0
            garment_lbs_weights = 0
            for i in tqdm(range(n_samples)):
                garment_shapedirs_sampled, garment_posedirs_sampled, garment_lbs_weights_sampled = sample_skinningweights(
                    garment_verts, smpl_tree, distances ** 0.5, body_model)
                garment_shapedirs += garment_shapedirs_sampled
                garment_posedirs += garment_posedirs_sampled
                garment_lbs_weights += garment_lbs_weights_sampled
            garment_shapedirs = torch.tensor(garment_shapedirs / n_samples)
            garment_posedirs = torch.tensor(garment_posedirs / n_samples)
            garment_lbs_weights = torch.tensor(garment_lbs_weights / n_samples)


        betas = smplx_dict['betas'][:, :10]
        betas = torch.tensor(betas).cuda().float()

        full_pose_list = []
        model_type = self.model_type
        if model_type == 'smplx':
            param_list = ['global_orient', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'left_hand_pose', 'right_hand_pose']
        elif model_type == 'smpl':
            param_list = ['global_orient', 'body_pose']
        else:
            raise ValueError(f'Unknown model type {model_type}')


        for k in param_list:
            full_pose_list.append(smplx_dict[k])
        full_pose = np.concatenate(full_pose_list, axis=1)
        full_pose = torch.tensor(full_pose).cuda().float()


        J_transformed, joint_transforms = my_lbs.get_transformed_joints(betas.cuda(), full_pose.cuda(), body_model.v_template.cuda(),
                                                                        body_model.shapedirs.cuda(),
                                                                        body_model.J_regressor.cuda(),
                                                                        body_model.parents.cuda())

        T = makeT(garment_lbs_weights.cuda(), J_transformed.cuda(), joint_transforms.cuda(), 1)[:, :, :3]

        garment_vertices_pt = torch.tensor(garment_verts).cuda().float()

        N = T.shape[1]
        ones = torch.ones(1, N, 1, 1).cuda()
        homo_delta = torch.matmul(T[..., -1:], ones)
        garment_vertices_md = garment_vertices_pt - homo_delta[0,...,0]

        T3 = T[..., :3]
        T3_inv = torch.inverse(T3)



        garment_vertices_unposed = torch.matmul(T3_inv, garment_vertices_md.unsqueeze(-1))[0, ..., 0]
        bs = blend_shapes(betas, garment_shapedirs.cuda())[0].detach()
        g_verts_unshaped = garment_vertices_unposed - bs

        return g_verts_unshaped.cpu().numpy()

def make_restpos_dict(vertices_full, faces_full):
    """
    Create a dictionary for a garment from an obj file
    """

    restpos_dict = dict()
    restpos_dict['rest_pos'] = vertices_full
    restpos_dict['faces'] = faces_full.astype(np.int64)
    restpos_dict['node_type'] = np.zeros_like(vertices_full[:, :1]).astype(np.int64)

    G = make_graph_from_faces(faces_full)
    components = list(nx.connected_components(G))
    garment_id = np.zeros_like(vertices_full[:, :1]).astype(np.int64)

    for i, component in enumerate(components):
        component = np.array(list(component))
        garment_id[component] = i

    restpos_dict['garment_id'] = garment_id

    return restpos_dict


def obj2template(obj_path, verbose=False):
    gc = GarmentCreator(None, None, None, None, collect_lbs=False, coarse=True, verbose=verbose, approximate_center=True)    
    out_dict = gc.make_garment_dict(obj_path)

    return out_dict



def add_coarse_edges(garment_dict, n_coarse_levels=4, approximate_center=True):
    n_levels = n_coarse_levels

    faces = garment_dict['faces']
    G = make_graph_from_faces(faces)

    components = list(nx.connected_components(G))

    cGd_list = []
    for component in components:
        cg_dict = dict()


        cG = G.subgraph(component)
        component_ids = np.array(list(component))
        faces_mask = np.isin(faces, component_ids).all(axis=1)

        faces_component = faces[faces_mask]

        if approximate_center:
            center_nodes = [approximate_graph_center(cG)]
        else:
            center_nodes = nx.center(cG)

        cg_dict['center'] = center_nodes
        cg_dict['coarse_edges'] = dict()

        for center in center_nodes[:3]:
            coarse_edges_dict = make_coarse_edges(faces_component, center, n_levels=n_levels)
            cg_dict['coarse_edges'][center] = coarse_edges_dict
        cGd_list.append(cg_dict)

    cGdk_list = [d['coarse_edges'].keys() for d in cGd_list]
    ctuples = list(itertools.product(*cGdk_list))


    center_list = []
    coarse_edges_dict = dict()
    for ci, ctuple in enumerate(ctuples):
        center_list.append(ci)
        coarse_edges_dict[ci] = dict()

        for l in range(n_levels):
            ce_list = []
            for i, d in enumerate(cGd_list):
                ce_list.append(d['coarse_edges'][ctuple[i]][l])

            ce_list = np.concatenate(ce_list, axis=0)

            coarse_edges_dict[ci][l] = ce_list


    garment_dict['center'] = np.array(center_list)
    garment_dict['coarse_edges'] = coarse_edges_dict

    return garment_dict




def create_smpl_pose_file(source_sequence_file, out_path, index):
    smpl_sequence = dict(np.load(source_sequence_file, allow_pickle=True))
    new_dict = {}

    new_dict['betas'] = smpl_sequence['betas'][None, :10]
    new_dict['transl'] = smpl_sequence['trans'][index:index+1]
    new_dict['global_orient'] = smpl_sequence['poses'][index:index+1, :3]
    new_dict['body_pose'] = smpl_sequence['poses'][index:index+1, 3:72]

    pickle_dump(new_dict, out_path)

def create_smplx_pose_file(source_sequence_file, out_path, index):
    smplx_sequence = dict(np.load(source_sequence_file, allow_pickle=True))
    new_dict = {}
    new_dict['betas'] = smplx_sequence['betas'][None, :10]

    new_dict['transl'] = smplx_sequence['trans'][index:index+1]
    new_dict['global_orient'] = smplx_sequence['root_orient'][index:index+1]
    new_dict['body_pose'] = smplx_sequence['pose_body'][index:index+1]
    new_dict['jaw_pose'] = smplx_sequence['pose_jaw'][index:index+1]
    new_dict['left_hand_pose'] = smplx_sequence['pose_hand'][index:index+1, :45]
    new_dict['right_hand_pose'] = smplx_sequence['pose_hand'][index:index+1, 45:]
    new_dict['leye_pose'] = smplx_sequence['pose_eye'][index:index+1, :3]
    new_dict['reye_pose'] = smplx_sequence['pose_eye'][index:index+1, 3:6]

    pickle_dump(new_dict, out_path)

