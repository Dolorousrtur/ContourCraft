from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import numpy as np
from utils.simulator import Simulator
import smplx

from utils.io import load_obj, save_obj, pickle_dump, pickle_load
from utils.common import copy_pyg_batch


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
    def __init__(self, checkpoint_path, garment_dict_file, n_epochs=2, n_steps_per_epoch=30):
        self.simulator = Simulator(checkpoint_path)
        self.garment_dict_file = Path(garment_dict_file)
        self.n_epochs = n_epochs
        self.n_steps_per_epoch = n_steps_per_epoch
        pass

    def untangle_single(self, untanglement, inner_garments, new_garment):
        for i in range(self.n_epochs):
            new_sample = untanglement.make_sample(inner_garments, [new_garment], n_steps=30)
            trajectories_dict = self.simulator._run_sequence(new_sample)
            untanglement.update_garments_from_trajectory(trajectories_dict, [new_garment])

            # if i == self.n_epochs - 1:
            #     break

            all_garments = inner_garments + [new_garment]
            new_sample = untanglement.make_sample([], all_garments, n_steps=self.n_steps_per_epoch)
            trajectories_dict = self.simulator._run_sequence(new_sample)
            untanglement.update_garments_from_trajectory(trajectories_dict, all_garments)

        out_path = '/home/agrigorev/Data/temp/untanglement_debug.pkl'
        pickle_dump(trajectories_dict, out_path)
        print(f'Sequence with the untanglement process saved to {out_path}')

        return untanglement

    def untangle_all(self, garment_list):
        garment_name_comma = ','.join(garment_list)
        sample = self.simulator._create_zeropos_sample(self.garment_dict_file, garment_name_comma, 
                                                       self.n_steps_per_epoch, add_bvh_data=True)
        untanglement = Untanglement(sample)

        n_garments = len(garment_list)
        for i in range(1, n_garments):
            inner_garments = garment_list[:i]
            new_garment = garment_list[i]
            untanglement = self.untangle_single(untanglement, inner_garments, new_garment)
        

        return untanglement

class Untanglement:
    def __init__(self, sample, use_uv=True):

        self.use_uv = use_uv
        self.sample = copy_pyg_batch(sample)
        self.garment_names = self.sample['garment_name'][0].split(',')

        self.garment_samples_dict = self._build_garments_dict(sample)
        self.body_dict = self._build_body_dict(sample)

        self.edge_types = self.sample.edge_types


    def create_updated_garment_dict(self, garment_creator, original_gdict_path, new_gdict_path):
        original_garments_dict = pickle_load(original_gdict_path)
        new_garments_dict = dict()

        for garment_name in self.garment_names:
            garment_dict = original_garments_dict[garment_name]
            garment_data = self.garment_samples_dict[garment_name]

            new_verts = garment_data['verts'].cpu().numpy()
            faces = garment_dict['faces']

            new_lbs_dict = garment_creator.make_lbs_dict(new_verts, faces)
            garment_dict['lbs'] = new_lbs_dict

            new_garments_dict[garment_name] = garment_dict

        pickle_dump(new_garments_dict, new_gdict_path)


    def _build_body_dict(self, sample):
        body_dict = dict()
        body_dict['verts'] = sample['obstacle'].pos[:, 0]
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
            verts_garment = sample['cloth'].pos[vertex_mask, 0]
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

            print('saved_verts', garment_data['verts'].shape)
            print('new_verts', new_verts.shape)
            garment_data['verts'] = new_verts
        

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
        pos = pos.unsqueeze(1).expand(-1, n_steps, -1)
        new_sample['obstacle'].prev_pos = pos.clone()
        new_sample['obstacle'].pos = pos.clone()
        new_sample['obstacle'].target_pos = pos.clone()
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
        pos = pos.unsqueeze(1).expand(-1, n_steps, -1)
        new_sample['cloth'].prev_pos = pos.clone()
        new_sample['cloth'].pos = pos.clone()
        new_sample['cloth'].target_pos = pos.clone()
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




class MGPreprocessor:
    def __init__(self, temp_dir, smplx_models_dir, gender, garment_dict_file=None):
        self.temp_dir = Path(temp_dir)

        wp.init()
        wp.set_device('cuda:0')

        self.smplx_model_path = Path(smplx_models_dir) / f'SMPLX_{gender.upper()}.npz'
        self.smplx_model = smplx.create(model_path=str(self.smplx_model_path), model_type='smplx', gender=gender, num_betas=10, use_pca=True, num_pca_comps=12)
        self.gender = gender

        self.garments_info = {}

        self.garment_dict_file = garment_dict_file
        self.runner = None
        self.config = None
        self.common_pose_path = None
        pass

    def _check_garment_dict_file(self, new_garment_dict_file):
        if self.garment_dict_file is None:
            if new_garment_dict_file is None:
                raise ValueError('garment_dict_file not set')
            else:
                self.garment_dict_file = new_garment_dict_file

    def _check_ginfo(self, garment_name, key, new_val):
        if new_val is not None:
            if garment_name not in self.garments_info:
                self.garments_info[garment_name] = dict()            
            self.garments_info[garment_name][key] = new_val


        if garment_name not in self.garments_info:
            if new_val is None:
                raise ValueError(f'{key} not set for new garment_name "{garment_name}"')
            else:
                self.garments_info[garment_name] = dict()
                self.garments_info[garment_name][key] = new_val
        elif key not in self.garments_info[garment_name]:
            if new_val is None:
                raise ValueError(f'{key} not set for garment_name "{garment_name}"')
            else:
                self.garments_info[garment_name][key] = new_val
              
    def _check_common_pose_path(self, new_pose_path):
        if self.common_pose_path is None:
            if new_pose_path is None:
                raise ValueError(f'pose_path not set')
            else:
                self.common_pose_path = new_pose_path

    def _check_runner(self):
        if self.runner is None or self.config is None:
            modules, config = load_params('ccraft')

            runner_module, runner, aux_modules = create_runner(modules, config, finetune=False)

            checkpoint_path = os.path.join(DEFAULTS.data_root, 'trained_models/ccraft.pth')
            state_dict =  torch.load(checkpoint_path)
            runner.load_state_dict(state_dict['training_module'])
            self.runner = runner
            self.config = config

    def _get_cano_verts(self, posed_garment_obj_path, pose_dict_path):
        pose_dict = pickle_load(pose_dict_path)
        garment_verts, garment_faces = load_obj(posed_garment_obj_path)
        # garment_verts = torch.tensor(garment_verts)

        cano_vertices, blend_weights, nb_idx = prepare_lbs(self.smplx_model, pose_dict, 
                                                           garment_verts-pose_dict['transl'], 
                                                           unpose=True, unshape=True)

        return cano_vertices, blend_weights, nb_idx, garment_faces
    
    def _make_sample(self, garment_name_list, garment_init_obj_path_list=None, n_frames=100):

        garment_name_list_str = ','.join(garment_name_list)

        if garment_init_obj_path_list is None:
            garment_init_obj_path_list_str = None
        else:
            garment_init_obj_path_list_str = ','.join(garment_init_obj_path_list)

        dataset_config = DatasetConfig(garment_dict_path=self.garment_dict_file, 
                                    garment_name_list=garment_name_list_str, 
                                    pose_path=None, 
                                    garment_init_obj_path_list=garment_init_obj_path_list_str, 
                                    smplx_model=self.smplx_model_path, 
                                    n_coarse_levels=4, n_frames=n_frames)
        dataset = create_dataset(dataset_config)
        dataloader_m = DataloaderModule(dataset, self.config['dataloader'])
        dataloader = dataloader_m.create_dataloader(is_eval=True)
        sample = next(iter(dataloader))
        return sample
    
    def unpose_garment(self, garment_obj_path, pose_path, garment_name=None, verbose=False):
        g1_unposed_v, _, _, g1_faces = self._get_cano_verts(garment_obj_path, pose_path)

        if garment_name is None:
            garment_name = Path(garment_obj_path).stem


        out_path = self.temp_dir / f'{garment_name}_unposed.obj'
        save_obj(out_path, g1_unposed_v, g1_faces)

        if verbose:
            print(f'Unposed "{garment_name}" saved to {out_path}')

        if garment_name not in self.garments_info:
            self.garments_info[garment_name] = dict()

        self.garments_info[garment_name]['init_obj'] = str(garment_obj_path)
        self.garments_info[garment_name]['pose_path'] = str(pose_path)
        self.garments_info[garment_name]['unposed_obj'] = str(out_path)

        self.common_pose_path = pose_path


        return g1_unposed_v, g1_faces
    
    def make_garment_dict(self, garment_name, garment_init_obj_path=None, garment_dict_file=None, verbose=False, pinned_verts=None):
        self._check_garment_dict_file(garment_dict_file)
        self._check_ginfo(garment_name, "init_obj", garment_init_obj_path)


        garment_init_obj_path = self.garments_info[garment_name]['init_obj']
        if not os.path.exists(self.garment_dict_file):
            garment_dict = {}
        else:
            garment_dict = pickle_load(self.garment_dict_file)
        gdict = make_garment_dict(garment_init_obj_path, None, training=False, verbose=verbose, pinned_verts=pinned_verts)

        garment_dict[garment_name] = gdict

        pickle_dump(garment_dict, self.garment_dict_file)

        if verbose:
            print(f'Garment dict for "{garment_name}" updated and saved to {self.garment_dict_file}')

    def simulate_zeropos(self, garment_name, n_frames=100, garment_dict_file=None, garment_unposed_obj_path=None, verbose=False):

        self._check_garment_dict_file(garment_dict_file)
        self._check_ginfo(garment_name, 'init_obj', garment_unposed_obj_path)
        self._check_runner()


        runner = self.runner
        garment_unposed_obj_path = self.garments_info[garment_name]['unposed_obj']

        
        # dataset_config = DatasetConfig(garment_dict_path=garment_dict_path, 
        #                             garment_name=garment_name, 
        #                             pose_path=pose_path, 
        #                             garment_init_obj_path=garment_unposed_obj_path, 
        #                             smplx_model=self.smplx_model_path, 
        #                             n_coarse_levels=4, n_frames=n_frames)
        # dataset = create_dataset(dataset_config)
        # dataloader_m = DataloaderModule(dataset, config['dataloader'])
        # dataloader = dataloader_m.create_dataloader(is_eval=True)
        # sample = next(iter(dataloader))

        sample = self._make_sample([garment_name], [garment_unposed_obj_path], n_frames)


        sample = runner.prepare_sample(sample)
        trajectories_dict = runner.valid_rollout(sample)


        verts = trajectories_dict['pred'][-1]
        faces = trajectories_dict['cloth_faces']

        out_path = os.path.join(self.temp_dir, garment_name+'_zerohood.obj')
        save_obj(out_path, verts, faces)

        if verbose:
            print(f'Zero pose simulation for "{garment_name}" saved to {out_path}')


        self.garments_info[garment_name]['zeropos_obj'] = out_path

        out_traj_path = os.path.join(self.temp_dir, 'traj', garment_name+'_zeropos_traj.pkl')
        pickle_dump(trajectories_dict, out_traj_path)

        if verbose:
            print(f'Trajectory  saved to {out_traj_path}')

        return verts, faces
    

    def _add_sample_as_obstacle(self, sample_outer, sample_inner):


        # print('sample_outer', sample_outer)
        # print('sample_inner', sample_inner)
        example_outer = sample_outer.get_example(0)
        example_inner = sample_inner.get_example(0)

        obstacle_pos = example_outer['obstacle']['pos']
        inner_pos = example_inner['cloth']['pos']
        n_verts_inner = inner_pos.shape[0]

        combined_pos = torch.cat([obstacle_pos, inner_pos], dim=0)

        for k in ['prev_pos', 'pos', 'target_pos']:
            example_outer['obstacle'][k] = combined_pos

        obstacle_faces = example_outer['obstacle']['faces_batch'].T
        inner_faces = example_inner['cloth']['faces_batch'].T
        faces_offset = obstacle_faces.max() + 1
        combined_faces = torch.cat([obstacle_faces, inner_faces+faces_offset], dim=0).T
        example_outer['obstacle']['faces_batch'] = combined_faces

        for k in ['vertex_type', 'vertex_level']:
            obstacle_val = example_outer['obstacle'][k]
            inner_val = obstacle_val[:1].expand(n_verts_inner, -1)
            combined_val = torch.cat([obstacle_val, inner_val], dim=0)
            example_outer['obstacle'][k] = combined_val


        garment_name_outer =  example_outer['garment_name']
        garment_name_inner = example_inner['garment_name']
        combined_garment_name = f"{garment_name_outer}_o:{garment_name_inner}"
        example_outer['garment_name'] = combined_garment_name


        sample_new = Batch.from_data_list([example_outer])

        # print('sample_new', sample_new)
        return sample_new

    def _combine_samples_inner_as_obstacle(self, sample_outer, samples_inner):
        sample_new = copy_pyg_batch(sample_outer)
        for sample_inner in samples_inner:
            sample_new = self._add_sample_as_obstacle(sample_new, sample_inner)
        return sample_new
    
    def _set_pos_to_samples(self, samples, trajectory_dict):
        pos = trajectory_dict['pred'][-1]
        garment_id = trajectory_dict['garment_id']
        device = samples[0]['cloth']['pos'].device
        n_steps = samples[0]['cloth']['pos'].shape[1]

        samples_new = []
        for i, sample in enumerate(samples):
            vertex_mask = (garment_id == i)[..., 0]
            pos_garment = pos[vertex_mask]

            example = sample.get_example(0)
            for k in ['prev_pos', 'pos', 'target_pos']:
                v = torch.FloatTensor(pos_garment).to(device)
                v = v.unsqueeze(1) # (n_verts, 3) -> (n_verts, 1, 3)
                v = v.expand(-1, n_steps, -1) # (n_verts, 1, 3) -> (n_verts, n_steps, 3)
                example['cloth'][k] = v
            sample_new = Batch.from_data_list([example])
            samples_new.append(sample_new)
        return samples_new
    
    def _add_sample_as_cloth(self, sample_to, sample_from):

        example_to = sample_to.get_example(0)
        example_from = sample_from.get_example(0)

        cloth_pos_to = example_to['cloth']['pos']
        cloth_pos_from = example_from['cloth']['pos']
        n_verts_from = cloth_pos_from.shape[0]
        combined_pos = torch.cat([cloth_pos_to, cloth_pos_from], dim=0)

        for k in ['prev_pos', 'pos', 'target_pos']:
            example_to['cloth'][k] = combined_pos

        rest_pos_to = example_to['cloth']['rest_pos']
        rest_pos_from = example_from['cloth']['rest_pos']
        combined_rest_pos = torch.cat([rest_pos_to, rest_pos_from], dim=0)
        example_to['cloth']['rest_pos'] = combined_rest_pos

        cloth_faces = example_to['cloth']['faces_batch'].T
        inner_faces = example_from['cloth']['faces_batch'].T
        faces_offset = cloth_faces.max() + 1
        combined_faces = torch.cat([cloth_faces, inner_faces+faces_offset], dim=0).T
        example_to['cloth']['faces_batch'] = combined_faces

        for k in ['vertex_type', 'vertex_level']:
            cloth_val = example_to['cloth'][k]
            inner_val = cloth_val[:1].expand(n_verts_from, -1)
            combined_val = torch.cat([cloth_val, inner_val], dim=0)
            example_to['cloth'][k] = combined_val

        garment_id_to = example_to['cloth']['garment_id']
        garment_id_from = example_from['cloth']['garment_id']
        garment_id_offset = garment_id_to.max() + 1
        combined_garment_id = torch.cat([garment_id_to, garment_id_from+garment_id_offset], dim=0)
        example_to['cloth']['garment_id'] = combined_garment_id


        for edge_type in example_to.edge_types:
            edge_index_to = example_to.edge_index_dict[edge_type]
            edge_index_from = example_from.edge_index_dict[edge_type]

            edge_index_from = edge_index_from + faces_offset
            edge_index_to = torch.cat([edge_index_to, edge_index_from], dim=1)
            example_to[edge_type].edge_index = edge_index_to

        
        garment_name_to =  example_to['garment_name']
        garment_name_from = example_from['garment_name']
        combined_garment_name = f"{garment_name_to}_{garment_name_from}"
        example_to['garment_name'] = combined_garment_name

        sample_new = Batch.from_data_list([example_to])
        return sample_new

    
    def _combine_samples_all_cloth(self, samples):
        sample_new = copy_pyg_batch(samples[0])
        for sample_to_add in samples[1:]:
            sample_new = self._add_sample_as_cloth(sample_new, sample_to_add)
        return sample_new
    
    def _untangle_step_as_obstacle(self, sample_outer, samples_inner):

        sample_inner_as_obstacle = self._combine_samples_inner_as_obstacle(sample_outer, samples_inner)
        trajectories_dict = self.runner.valid_rollout(sample_inner_as_obstacle)
        sample_outer = self._set_pos_to_samples([sample_outer], trajectories_dict)[0]
        return sample_outer, trajectories_dict
    
    def _untangle_step_all_cloth(self, sample_outer, samples_inner):
        sample_all_cloth = self._combine_samples_all_cloth(samples_inner + [sample_outer])
        trajectories_dict = self.runner.valid_rollout(sample_all_cloth)
        samples_upd = self._set_pos_to_samples(samples_inner + [sample_outer], trajectories_dict)
        samples_inner = samples_upd[:-1]
        sample_outer = samples_upd[-1]
        return sample_outer, samples_inner, trajectories_dict

    def _convert_trajectory_to_all_cloth(self, trajectory_as_obstacle, trajectory_all_cloth):
        as_obstacle_opos = trajectory_as_obstacle['obstacle']

        n_obstacle_verts = trajectory_all_cloth['obstacle'].shape[1]

        cpos_inner = as_obstacle_opos[:, n_obstacle_verts:]
        cpos_outer = trajectory_as_obstacle['pred']


        cpos_new = np.concatenate([cpos_inner, cpos_outer], axis=1)

        opos_new = as_obstacle_opos[:, :n_obstacle_verts]

        trajectory_new = {}
        for k, v in trajectory_all_cloth.items():
            if k == 'obstacle':
                trajectory_new[k] = opos_new
            elif k == 'pred':
                trajectory_new[k] = cpos_new
            else:
                trajectory_new[k] = v


        return trajectory_new
    
    def _combine_trajectories(self, trajectories_dict_list):
        keys = trajectories_dict_list[0].keys()

        out_dict = {}
        for k in keys:
            if k in ['pred', 'obstacle']:
                v_list = [td[k] for td in trajectories_dict_list]
                v_combined = np.concatenate(v_list, axis=0)
            else:
                v_combined = trajectories_dict_list[0][k]

            out_dict[k] = v_combined

        return out_dict
    
    def _untangle_epoch(self, sample_outer, samples_inner):


        sample_outer, trajectories_dict_o = self._untangle_step_as_obstacle(sample_outer, samples_inner)


        sample_outer, samples_inner, trajectories_dict_c = self._untangle_step_all_cloth(sample_outer, samples_inner)

        trajectories_dict_o = self._convert_trajectory_to_all_cloth(trajectories_dict_o, trajectories_dict_c)

        trajectories_dict_combined = self._combine_trajectories([trajectories_dict_o, trajectories_dict_c])

        return sample_outer, samples_inner, trajectories_dict_combined

    def _untangle_one_garment(self, sample_outer, samples_inner, outer_garment_name, inner_garment_names, n_epochs):

        out_garment_name = f"{outer_garment_name}_{','.join(inner_garment_names)}"

        trajectories_dict_list = []
        for i in range(n_epochs):
            sample_outer, samples_inner, trajectories_dict = self._untangle_epoch(sample_outer, samples_inner)
            trajectories_dict_list.append(trajectories_dict)

        trajectories_dict_combined = self._combine_trajectories(trajectories_dict_list)
        out_traj_path = os.path.join(self.temp_dir, 'traj', f"{out_garment_name}_untangle_traj.pkl")
        pickle_dump(trajectories_dict_combined, out_traj_path)
        print(out_traj_path)

        return sample_outer, samples_inner
    
    def update_garment_dict_with_outfit(self, garment_dict_file, sample_dict, outfit_name, verbose=False):

        garment_dict = pickle_load(garment_dict_file)

        for gname, sample in sample_dict.items():
            gdict  = deepcopy(garment_dict[gname])
            gdict.pop('lbs')

            garment_verts = sample['cloth']['pos'][:, -1].cpu().numpy()
            garment_faces = sample['cloth']['faces_batch'].T.cpu().numpy()


            lbs_dict = make_garment_template_from_obj(garment_verts, garment_faces, self.smplx_model_path, model_type='smplx', n_samples=1000)
            gdict['lbs'] = lbs_dict

            new_gname = f"{outfit_name}::{gname}"
            garment_dict[new_gname] = gdict

            if verbose:
                print(f'Added new garment dict for {new_gname} to {garment_dict_file}')

        pickle_dump(garment_dict, garment_dict_file)

    
    def untangle_outfit(self, garment_names_ordered, outfit_name, n_epochs=2, n_frames_per_epoch=100, 
                 garment_dict_file=None, verbose=False):
        
        self._check_garment_dict_file(garment_dict_file)
        self._check_runner()

        sample_dict = {}
        for gname in garment_names_ordered:
            sample = self._make_sample([gname], None, n_frames_per_epoch)
            sample_dict[gname] = sample


        for i in range(1, len(garment_names_ordered)):
            outer_gname = garment_names_ordered[i]
            inner_gnames = garment_names_ordered[:i]

            sample_outer = sample_dict[outer_gname]
            samples_inner = [sample_dict[ign] for ign in inner_gnames]

            sample_outer, samples_inner = self._untangle_one_garment(sample_outer, samples_inner,  outer_gname, inner_gnames, n_epochs)

            sample_dict[outer_gname] = sample_outer
            for ign, sample_inner in zip(inner_gnames, samples_inner):
                sample_dict[ign] = sample_inner


        self.update_garment_dict_with_outfit(self.garment_dict_file, sample_dict, outfit_name, verbose=verbose)


        # trajectories_dict = self.runner.valid_rollout(sample)

        # verts = trajectories_dict['pred'][-1]
        # faces = trajectories_dict['cloth_faces']
        # out_path = os.path.join(self.temp_dir, outer_garment_name+'_untangle.obj')
        # save_obj(out_path, verts, faces)

        # if verbose:
        #     print(f'Zero pose simulation for "{outer_garment_name}" saved to {out_path}')

        # out_traj_path = os.path.join(self.temp_dir, 'traj', outer_garment_name+'_untangle_traj.pkl')
        # pickle_dump(trajectories_dict, out_traj_path)

        # if verbose:
        #     print(f'Trajectory  saved to {out_traj_path}')      

        
        # self.garments_info[outer_garment_name]['untangle_obj'] = out_path
        # for ign in inner_garment_name_list:
        #     self.garments_info[ign]['untangle_obj'] = self.garments_info[ign]['zeropos_obj']


    def resimulate(self, garment_name_list, n_frames=30, garment_dict_file=None, garment_untangle_obj_list=None, verbose=False):
        
        self._check_garment_dict_file(garment_dict_file)

        if garment_untangle_obj_list is None:
            garment_untangle_obj_list = [None]*len(garment_name_list)

        untangle_obj_list = []
        for ign, ig in zip(garment_name_list, garment_untangle_obj_list):
            self._check_ginfo(ign, 'untangle_obj', ig)
            untangle_obj_list.append(self.garments_info[ign]['untangle_obj'])

        self._check_runner()


        sample = self._make_sample(garment_name_list, untangle_obj_list, n_frames)
        trajectories_dict = self.runner.valid_rollout(sample)

        verts = trajectories_dict['pred'][-1]
        faces = trajectories_dict['cloth_faces']
        garment_id = trajectories_dict['garment_id']

        face_shift = 0
        for  i in range(garment_id.max()+1):
            garment_id_mask = (garment_id == i)[..., 0]
            garment_verts = verts[garment_id_mask]
            faces_mask = garment_id_mask[faces].all(axis=1)

            garment_faces = faces[faces_mask]
            face_shift_new = garment_faces.max() + 1

            garment_faces = garment_faces - face_shift
            face_shift = face_shift_new

            garment_name = garment_name_list[i]

            out_path = os.path.join(self.temp_dir, garment_name+'_final.obj')
            save_obj(out_path, garment_verts, garment_faces)

            if verbose:
                print(f'Final geometry for "{garment_name}" saved to {out_path}')

            self.garments_info[garment_name]['final_obj'] = out_path

        out_traj_path = os.path.join(self.temp_dir, 'traj', '_'.join(garment_name_list)+'_resimul.pkl')
        pickle_dump(trajectories_dict, out_traj_path)

        if verbose:
            print(f'Trajectory  saved to {out_traj_path}')  


    def update_gdict_with_lbs(self, garment_name, garment_dict_file=None, garment_final_obj=None, verbose=False):
        self._check_garment_dict_file(garment_dict_file)

        self._check_ginfo(garment_name, 'zeropos_obj', garment_final_obj)

        garment_dict = pickle_load(self.garment_dict_file)
        final_obj_path = self.garments_info[garment_name]['zeropos_obj']

        lbs_dict = make_garment_template_from_obj(final_obj_path, self.smplx_model_path, model_type='smplx', n_samples=1000)
        garment_dict[garment_name]['lbs'] = lbs_dict


        pickle_dump(garment_dict, self.garment_dict_file)
        if verbose:
            print(f'Garment dict for {garment_name} updated and saved to {self.garment_dict_file}')


    def add_garment(self, garment_name, garment_obj_path, pose_path, verbose=False, pinned_verts=None):
        # 1. Unpose
        if verbose:
            print(f'\n\n1. {garment_name}: Unposing the garment...')

        self.unpose_garment(garment_obj_path, pose_path, garment_name, verbose=verbose)

        # 2. Make garment dict
        if verbose:
            print(f'\n\n2. {garment_name}: Making a garment dict ...')
        self.make_garment_dict(garment_name, verbose=verbose, pinned_verts=pinned_verts)


        # 3. Simulate zeropos
        if verbose:
            print(f'\n\n3. Simulating in zero pose and shape...')
        self.simulate_zeropos(garment_name, verbose=verbose)


        # 4. Update garment dict with lbs
        if verbose:
            print(f'\n\n4. {garment_name}: Updating garment dict with lbs weights...')
        self.update_gdict_with_lbs(garment_name, verbose=verbose)


    def compose_outfit(self, garment_name_list, verbose=False):
        pass

        # 4. Untangle
        if verbose:
            print('\n\n4. Untangling garments...')
        for i, input_gdict_outer in enumerate(garment_name_list):
            if i == 0:
                continue

            outer_gname = input_gdict_outer['name']
            inner_glist = [input_gdict['name'] for input_gdict in garment_name_list[:i]]

            if verbose:
                print()
                print(f"Inner garments: {inner_glist}")
                print(f"Outer garment: {outer_gname}")

            self.untangle_outfit(outer_gname, inner_glist, verbose=verbose)


    def process_full(self, input_dict_list, verbose=False):

        # 1. Unpose
        if verbose:
            print('\n\n1. Unposing garments...')

        for input_gdict in input_dict_list:
            self.unpose_garment(input_gdict['init_obj'], input_gdict['pose_path'], input_gdict['name'], verbose=verbose)

        # 2. Make garment dict
        if verbose:
            print('\n\n2. Making garment dict...')
        for input_gdict in input_dict_list:
            gname = input_gdict['name']
            self.make_garment_dict(gname, verbose=verbose)

        # 3. Simulate zeropos
        if verbose:
            print('\n\n3. Simulating garments in zero pose and shape...')
        for input_gdict in input_dict_list:
            gname = input_gdict['name']
            self.simulate_zeropos(gname, verbose=verbose)

        # 4. Untangle
        if verbose:
            print('\n\n4. Untangling garments...')
        for i, input_gdict_outer in enumerate(input_dict_list):
            if i == 0:
                continue

            outer_gname = input_gdict_outer['name']
            inner_glist = [input_gdict['name'] for input_gdict in input_dict_list[:i]]

            if verbose:
                print()
                print(f"Inner garments: {inner_glist}")
                print(f"Outer garment: {outer_gname}")

            self.untangle_outfit(outer_gname, inner_glist, verbose=verbose)


        # 5. Resimulate
        if verbose:
            print('\n\n5. Resimulating all garments together in zero pose/shape...')
        gname_list = [input_gdict['name'] for input_gdict in input_dict_list]
        self.resimulate(gname_list, verbose=verbose)


        # 6. Update garment dict with lbs
        if verbose:
            print('\n\n6. Updating garment dict with lbs weights...')
        for input_gdict in input_dict_list:
            gname = input_gdict['name']
            self.update_gdict_with_lbs([gname], verbose=verbose)


        if verbose:
            print('\n\nProcessing finished!')
            print(f"Garments {gname_list} are stored in garment_dict file: {self.garment_dict_file}")

def add_mesh_to_obstacle(sample, mesh_obj_path):
    new_verts, new_faces = load_obj(mesh_obj_path)
    new_verts = torch.FloatTensor(new_verts)
    new_faces = torch.LongTensor(new_faces)

    sample_graph = sample.get_example(0)
    n_frames = sample_graph['obstacle'].prev_pos.shape[1]

    new_n_verts = new_verts.shape[0]

    new_pos = torch.stack([new_verts]*n_frames, dim=1)
    for k in ['prev_pos', 'pos', 'target_pos']:
        sample_graph['obstacle'][k] = torch.cat([sample_graph['obstacle'][k], new_pos], dim=0)

    new_vertex_type = torch.ones((new_n_verts,1), dtype=torch.long)
    sample_graph['obstacle']['vertex_type'] = torch.cat([sample_graph['obstacle']['vertex_type'], new_vertex_type], dim=0)

    new_vertex_level = torch.zeros((new_n_verts,1), dtype=torch.long)
    sample_graph['obstacle']['vertex_level'] = torch.cat([sample_graph['obstacle']['vertex_level'], new_vertex_level], dim=0)

    faces_offset = sample_graph['obstacle'].faces_batch.max() + 1
    new_faces = new_faces + faces_offset
    sample_graph['obstacle']['faces_batch'] = torch.cat([sample_graph['obstacle']['faces_batch'], new_faces.T], dim=1)

    sample_new = Batch.from_data_list([sample_graph])
    return sample_new