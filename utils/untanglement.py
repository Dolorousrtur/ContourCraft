from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import numpy as np
from utils.mesh_creation import GarmentCreator
from utils.simulator import Simulator
import smplx

from utils.io import load_obj, save_obj, pickle_dump, pickle_load
from utils.common import copy_pyg_batch, move2device


from torch_geometric.data import Batch, HeteroData
from utils.arguments import load_params, create_runner
from utils.arguments import load_params
from utils.defaults import DEFAULTS
from pathlib import Path
import torch
import os
import warp as wp

# from ccraft.datasets.zeropos_init import Config as DatasetConfig
# from ccraft.datasets.zeropos_init import create as create_dataset
# from ccraft.utils.dataloader import DataloaderModule



class Untangler:
    def __init__(self, garment_creator, checkpoint_path, n_epochs=2, n_steps_per_stage=30, gender='female', use_uv=False, **kwargs):
        self.simulator = Simulator(checkpoint_path)
        # self.garment_dict_file = Path(garment_dict_file)
        self.n_epochs = n_epochs
        self.n_steps_per_epoch = n_steps_per_stage
        self.gender = gender
        self.use_uv = use_uv
        self.data_kwargs = kwargs

        self.garment_creator = garment_creator
        self.gc = GarmentCreator(None, None, None, collect_lbs=False, coarse=True, verbose=False)   

    def untangle_single(self, untanglement, inner_garments, new_garment, trajectories_list=None):
        if trajectories_list is None:
            trajectories_list = []
        for i in range(self.n_epochs):
            print(f'\nUntangling {new_garment} with {inner_garments} ({i+1}/{self.n_epochs})')
            print(f'Simulating inner garments as obstacles...')
            # new_sample = untanglement.make_sample(inner_garments, [new_garment], n_steps=30)
            new_sample = untanglement.make_sample(inner_garments, [new_garment], n_steps=self.n_steps_per_epoch)
            trajectories_dict = self.simulator._run_sequence(new_sample)
            untanglement.update_garments_from_trajectory(trajectories_dict, [new_garment])

            trajectories_dict = untanglement._fix_trajectory(trajectories_dict, inner_garments)
            trajectories_list.append(trajectories_dict)

            print(f'Simulating all garments as cloth...')
            all_garments = inner_garments + [new_garment]
            new_sample = untanglement.make_sample([], all_garments, n_steps=self.n_steps_per_epoch)
            trajectories_dict = self.simulator._run_sequence(new_sample)
            trajectories_list.append(trajectories_dict)
            untanglement.update_garments_from_trajectory(trajectories_dict, all_garments)


        return untanglement, trajectories_list

    def untangle_all(self, garment_list, outfit_name):
        garment_name_comma = ','.join(garment_list)
        sample = self.simulator._create_zeropos_sample(garment_name_comma, 
                                                       self.n_steps_per_epoch, gender=self.gender, **self.data_kwargs)
        sample = move2device(sample, 'cuda:0')
        untanglement = Untanglement(sample, use_uv=self.use_uv)

        n_garments = len(garment_list)
        trajectories_list = []
        for i in range(1, n_garments):
            inner_garments = garment_list[:i]
            new_garment = garment_list[i]
            untanglement, trajectories_list = self.untangle_single(untanglement, inner_garments, new_garment, trajectories_list=trajectories_list)
        
        combined_trajectory_dict = untanglement._combine_trajectories(trajectories_list)

        untanglement.save_new_garment_dicts(self.garment_creator, outfit_name, **self.data_kwargs)

        return untanglement, combined_trajectory_dict

class Untanglement:
    def __init__(self, sample, use_uv=True):

        self.use_uv = use_uv
        self.sample = copy_pyg_batch(sample)
        self.garment_names = self.sample['garment_name'][0].split(',')

        self.garment_samples_dict = self._build_garments_dict(sample)
        self.body_dict = self._build_body_dict(sample)

        self.edge_types = self.sample.edge_types

    def save_new_garment_dicts(self, garment_creator, outfit_name, **kwargs):

        if 'garment_dicts_dir' in kwargs:
            original_garments_root = Path(kwargs['garment_dicts_dir'])
        else:
            original_garments_root = Path(DEFAULTS.aux_data) / 'garment_dicts'

        for garment_name in self.garment_names:
            original_garments_dict_path = original_garments_root / f'{garment_name}.pkl'
            garment_dict = pickle_load(original_garments_dict_path)
            garment_data = self.garment_samples_dict[garment_name]

            new_verts = garment_data['verts'].cpu().numpy()
            faces = garment_dict['faces']

            new_lbs_dict = garment_creator.make_lbs_dict(new_verts, faces)
            garment_dict['lbs'] = new_lbs_dict

            out_path = original_garments_root / outfit_name / f'{garment_name}.pkl'
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pickle_dump(garment_dict, out_path)

            print(f'Saving the untangled version of {garment_name} to {out_path}')

        gnames_w_outfit = [f"{outfit_name}/{gn}" for gn in self.garment_names]
        gnames_w_outfit_str = ','.join(gnames_w_outfit)

        print(f'\nYou can now simulate the outfit with either:\n'
                f'garment_name="{gnames_w_outfit_str}"\n'
                f'or simply:\n'
                f'garment_name="{outfit_name}"\n')

    def _build_body_dict(self, sample):
        body_dict = dict()
        body_dict['verts'] = sample['obstacle'].pos
        body_dict['vertex_type'] = sample['obstacle'].vertex_type
        body_dict['vertex_level'] = sample['obstacle'].vertex_level
        body_dict['faces'] = sample['obstacle'].faces_batch.T 

        return body_dict


    def _build_garments_dict(self, sample):
        garment_names = self.garment_names

        garment_id = sample['cloth'].garment_id
        faces = sample['cloth'].faces_batch.T

        if self.use_uv:
            uv_faces = sample['cloth'].uv_faces_batch.T
            uv_coords_offset = 0

        garments_dict = {}

        vertex_offset = 0
        for i, garment_name in enumerate(garment_names):
            garment_data = dict()

            vertex_mask = (garment_id == i)[..., 0]
            verts_garment = sample['cloth'].pos[vertex_mask]
            vertex_type = sample['cloth'].vertex_type[vertex_mask]
            vertex_level = sample['cloth'].vertex_level[vertex_mask]
            rest_pos = sample['cloth'].rest_pos[vertex_mask]

            garment_data['verts'] = verts_garment
            garment_data['rest_pos'] = rest_pos
            garment_data['vertex_type'] = vertex_type
            garment_data['vertex_level'] = vertex_level


            faces_mask = vertex_mask[faces].all(axis=1)
            faces_garment = faces[faces_mask]
            garment_data['faces'] = faces_garment - vertex_offset

            if self.use_uv:
                uv_faces_garment = uv_faces[faces_mask]
                uv_coords_inds_garment = torch.unique(uv_faces_garment)
                uv_coords_garment = sample['cloth'].uv_coords[uv_coords_inds_garment]

                uv_coords_inds_garment_sort = torch.argsort(uv_coords_inds_garment)
                uv_coords_garment = uv_coords_garment[uv_coords_inds_garment_sort]

                uv_faces_garment = uv_faces_garment - uv_coords_offset

                garment_data['uv_coords'] = uv_coords_garment
                garment_data['uv_faces'] = uv_faces_garment

                uv_coords_offset += uv_coords_garment.shape[0]
                


            garment_data['edges'] = dict()
            for edge_type in sample.edge_types:
                edge_index = sample[edge_type].edge_index.T
                edge_mask = vertex_mask[edge_index].all(axis=1)
                edge_index_garment = edge_index[edge_mask]
                garment_data['edges'][edge_type] = edge_index_garment - vertex_offset
            vertex_offset += verts_garment.shape[0]

            garments_dict[garment_name] = garment_data
        return garments_dict


    def make_sample(self, obstacle_garments, cloth_garments, n_steps=100):
        new_sample = copy_pyg_batch(self.sample)

        new_sample = HeteroData()
        new_sample = self.build_obstacle(new_sample, obstacle_garments, use_body=True, n_steps=n_steps)
        new_sample = self.build_cloth(new_sample, cloth_garments, n_steps=n_steps)

        new_sample.garment_name = f"{','.join(obstacle_garments)}_o:{','.join(cloth_garments)}"

        new_sample = Batch.from_data_list([new_sample])

        return new_sample
    
    def update_garments_from_trajectory(self, trajectory_dict, garment_list, idx=-1):
        garment_verts = trajectory_dict['pred'][idx]
        garment_id = trajectory_dict['garment_id'][:, 0]

        for i in range(len(garment_list)):
            garment_name = garment_list[i]
            garment_data = self.garment_samples_dict[garment_name]

            old_verts = garment_data['verts']
            new_verts = garment_verts[garment_id == i]
            new_verts = torch.FloatTensor(new_verts).to(old_verts.device)

            garment_data['verts'] = new_verts
    
    def _combine_trajectories(self, trajectory_dict_list):
        combined_trajectory_dict = defaultdict(list)
        obstacle_faces = trajectory_dict_list[0]['obstacle_faces']
        combined_trajectory_dict['obstacle_faces'] = obstacle_faces

        n_verts_list = [trajectory_dict['pred'].shape[1] for trajectory_dict in trajectory_dict_list]
        max_cloth_verts_id = np.argmax(n_verts_list)
        max_cloth_verts = trajectory_dict_list[max_cloth_verts_id]['pred'].shape[1]
        full_cloth_faces = trajectory_dict_list[max_cloth_verts_id]['cloth_faces']
        full_garment_id = trajectory_dict_list[max_cloth_verts_id]['garment_id']

        combined_trajectory_dict['cloth_faces'] = full_cloth_faces
        combined_trajectory_dict['garment_id'] = full_garment_id

        for trajectory_dict in trajectory_dict_list:
            obstacle_verts = trajectory_dict['obstacle']
            combined_trajectory_dict['obstacle'].append(obstacle_verts)

            cloth_verts = trajectory_dict['pred']
            n_cloth_verts = cloth_verts.shape[1]
            to_add = max_cloth_verts - n_cloth_verts

            extention = np.zeros((cloth_verts.shape[0], to_add, cloth_verts.shape[2]))
            cloth_verts = np.concatenate((cloth_verts, extention), axis=1)
            combined_trajectory_dict['pred'].append(cloth_verts)

        for k in ['obstacle', 'pred']:
            combined_trajectory_dict[k] = np.concatenate(combined_trajectory_dict[k], axis=0)
        return combined_trajectory_dict

    def _fix_trajectory(self, trajectory_dict, obstacle_garments):
        n_obstacle_verts = self.body_dict['verts'].shape[0]
        n_obstacle_faces = self.body_dict['faces'].shape[0]

        obstacle_verts_out = trajectory_dict['obstacle'][:, :n_obstacle_verts]
        obstacle_faces_out = trajectory_dict['obstacle_faces'][:n_obstacle_faces]

        vertex_offset = n_obstacle_verts
        face_offset = n_obstacle_faces


        garment_verts_out = []
        garment_faces_out = []
        garment_id_out = []

        garment_id_offset = 0
        for og in obstacle_garments:
            gdict = self.garment_samples_dict[og]
            n_verts = gdict['verts'].shape[0]
            n_faces = gdict['faces'].shape[0]

            garment_verts = trajectory_dict['obstacle'][:, vertex_offset:vertex_offset +  n_verts]
            garment_faces = trajectory_dict['obstacle_faces'][face_offset:face_offset + n_faces]
            garment_faces -= n_obstacle_verts


            garment_id = np.ones_like(garment_verts[0, :, :1], dtype=np.int32) * garment_id_offset

            garment_verts_out.append(garment_verts)
            garment_faces_out.append(garment_faces)
            garment_id_out.append(garment_id)

            garment_id_offset += 1
            vertex_offset += n_verts
            face_offset += n_faces

        garment_verts_orig = trajectory_dict['pred']
        garment_faces_orig = trajectory_dict['cloth_faces']



        garment_faces_orig += (vertex_offset - n_obstacle_verts)

        garment_id_orig = trajectory_dict['garment_id']
        garment_id_orig += garment_id_offset

        garment_verts_out.append(garment_verts_orig)
        garment_faces_out.append(garment_faces_orig)
        garment_id_out.append(garment_id_orig)

        garment_verts_out = np.concatenate(garment_verts_out, axis=1)
        garment_faces_out = np.concatenate(garment_faces_out, axis=0)
        garment_id_out = np.concatenate(garment_id_out, axis=0)

        out_trajectory_dict = dict()
        out_trajectory_dict['obstacle'] = obstacle_verts_out
        out_trajectory_dict['obstacle_faces'] = obstacle_faces_out
        out_trajectory_dict['pred'] = garment_verts_out
        out_trajectory_dict['cloth_faces'] = garment_faces_out
        out_trajectory_dict['garment_id'] = garment_id_out

        return out_trajectory_dict


    def build_obstacle(self, new_sample, obstacle_garments, use_body=True, n_steps=100):

        verts = []
        vertex_type = []
        vertex_level = []
        faces = []

        vertex_offset = 0
        if use_body:
            verts.append(self.body_dict['verts'])
            vertex_type.append(self.body_dict['vertex_type'])
            vertex_level.append(self.body_dict['vertex_level'])
            faces.append(self.body_dict['faces'])

            vertex_offset += self.body_dict['verts'].shape[0]

        for og in obstacle_garments:
            verts.append(self.garment_samples_dict[og]['verts'])
            vertex_type.append(self.garment_samples_dict[og]['vertex_type'])
            vertex_level.append(self.garment_samples_dict[og]['vertex_level'])
            faces.append(self.garment_samples_dict[og]['faces'] + vertex_offset)

            vertex_offset += self.garment_samples_dict[og]['verts'].shape[0]


        pos  = torch.cat(verts, dim=0)

        lookup = pos.unsqueeze(1).expand(-1, n_steps, -1)
        new_sample['obstacle'].prev_pos = pos.clone()
        new_sample['obstacle'].pos = pos.clone()
        new_sample['obstacle'].target_pos = pos.clone()
        new_sample['obstacle'].lookup = lookup.clone()
        new_sample['obstacle'].vertex_type = torch.cat(vertex_type, dim=0)
        new_sample['obstacle'].vertex_level = torch.cat(vertex_level, dim=0)
        new_sample['obstacle'].faces_batch = torch.cat(faces, dim=0).T

        
        return new_sample

    def build_cloth(self, new_sample, cloth_garments, n_steps=100):
        verts = []
        rest_pos = []
        vertex_type = []
        vertex_level = []
        faces = []
        garment_id = []

        if self.use_uv:
            uv_coords = []
            uv_faces = []
            vertex_uv_offset = 0

        edge_indices = defaultdict(list)

        vertex_offset = 0
        for i, cg in enumerate(cloth_garments):
            gdict = self.garment_samples_dict[cg]

            verts.append(gdict['verts'])
            rest_pos.append(gdict['rest_pos'])
            vertex_type.append(gdict['vertex_type'])
            vertex_level.append(gdict['vertex_level'])
            faces.append(gdict['faces'] + vertex_offset)

            if self.use_uv:
                uv_coords.append(gdict['uv_coords'])
                uv_faces.append(gdict['uv_faces'] + vertex_uv_offset)
                vertex_uv_offset += gdict['uv_coords'].shape[0]

            N_verts = gdict['verts'].shape[0]
            gid = torch.ones_like(gdict['vertex_type']) * i
            garment_id.append(gid)

            for edge_type in gdict['edges']:
                edge_indices[edge_type].append(gdict['edges'][edge_type] + vertex_offset)

            vertex_offset += gdict['verts'].shape[0]


        pos = torch.cat(verts, dim=0)
        lookup = pos.unsqueeze(1).expand(-1, n_steps, -1)
        new_sample['cloth'].prev_pos = pos.clone()
        new_sample['cloth'].pos = pos.clone()
        new_sample['cloth'].target_pos = pos.clone()
        new_sample['cloth'].lookup = lookup.clone()
        new_sample['cloth'].rest_pos = torch.cat(rest_pos, dim=0)
        new_sample['cloth'].vertex_type = torch.cat(vertex_type, dim=0)
        new_sample['cloth'].vertex_level = torch.cat(vertex_level, dim=0)
        new_sample['cloth'].garment_id = torch.cat(garment_id, dim=0)
        new_sample['cloth'].faces_batch = torch.cat(faces, dim=0).T

        if self.use_uv:
            new_sample['cloth'].uv_coords = torch.cat(uv_coords, dim=0)
            new_sample['cloth'].uv_faces_batch = torch.cat(uv_faces, dim=0)


        for edge_type in edge_indices:
            new_sample[edge_type].edge_index = torch.cat(edge_indices[edge_type], dim=0).T

        return new_sample



