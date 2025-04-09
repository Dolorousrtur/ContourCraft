import os
import pickle
import time

import numpy as np
import torch
import cccollisions
import torch_scatter
from torch_geometric.data import Batch

from utils.common import add_field_to_pyg_batch
from utils.io import pickle_dump
from utils.defaults import DEFAULTS
from utils.selfcollisions import CollisionHelper, find_close_faces, get_continuous_collisions

class CollisionSolver:
    def __init__(self, mcfg):
        self.collision_helper = CollisionHelper(mcfg.device)
        self.mcfg = mcfg

    @staticmethod
    def compute_riz_deltas(riz_ids, pinned_mask, mass, curr_pos, dx):
        V = riz_ids.shape[0]
        device = riz_ids.device
        riz_pinned = pinned_mask[riz_ids]

        masses = mass[riz_ids]
        velocities = dx[riz_ids]
        positions_t0 = curr_pos[riz_ids]

        # 3. Compute center of mass and average velocity
        mass_weighted_positions = masses * positions_t0
        total_mass = torch.sum(masses)
        center_of_mass = torch.sum(mass_weighted_positions, dim=0) / total_mass

        mass_weighted_velocities = masses * velocities
        average_velocity = torch.sum(mass_weighted_velocities, dim=0) / total_mass

        # Compute the inertia tensor
        positions_relative_to_com = positions_t0 - center_of_mass
        # I = torch.zeros(3, 3).to(device).double()
        mass_weighted_square_distances = positions_relative_to_com.pow(2).sum(1) * masses[:, 0]
        eye = torch.eye(3).to(positions_t0.device, positions_t0.dtype)
        mass_weighted_square_distances = mass_weighted_square_distances[:, None, None] * eye[None]
        posmass = torch.einsum(
            'ij,ik->ijk', positions_relative_to_com, positions_relative_to_com) * masses.view(-1, 1, 1)
        I = (mass_weighted_square_distances - posmass).sum(dim=0)

        # Compute the angular momentum
        angular_momentum = torch.cross(mass_weighted_velocities, positions_relative_to_com).sum(dim=0)

        new_angular_velocities = torch.linalg.solve(I.unsqueeze(0), angular_momentum.unsqueeze(0))
        new_angular_velocities = new_angular_velocities.squeeze()  # omega

        angular_velocity_magnitude = torch.norm(new_angular_velocities)
        rotation_axis = new_angular_velocities / angular_velocity_magnitude
        angle = angular_velocity_magnitude

        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)

        dot_product = rotation_axis[None,] * positions_relative_to_com

        term1 = positions_relative_to_com * cos_angle
        term2 = torch.cross(rotation_axis[None,], positions_relative_to_com) * sin_angle
        term3 = rotation_axis[None,] * dot_product * (1 - cos_angle)

        positions_relative_to_com_rotated = term1 + term2 + term3
        new_positions_t1 = center_of_mass + positions_relative_to_com_rotated + average_velocity

        rigid_dx = new_positions_t1 - positions_t0
        return rigid_dx



    def check_nan(self, tensor, a=False):
        is_nan = (tensor != tensor).any()
        if a:
            assert not is_nan
        else:
            if is_nan:
                print('NAN')

    def get_faces(self, state, use_cutout=True):
        faces = state['cloth'].faces_batch.T
        if use_cutout and 'faces_cutout_mask_batch' in state['cloth']:
            faces_cutout_mask = state['cloth'].faces_cutout_mask_batch[0]
            faces = faces[faces_cutout_mask]
        return faces

    def safecheck_RIZ(self, state, metrics_dict=None, label=''):
        faces = self.get_faces(state)

        verts0 = state['cloth'].pos.clone()
        verts1 = state['cloth'].pred_pos.clone()
        velocity = state['cloth'].pred_velocity.clone()
        timestep = state['cloth'].timestep


        velocity_dx = velocity

        mass = state['cloth'].v_mass.clone()
        dx = verts1 - verts0
        edges = None
        if 'penetrating_mask' in state['cloth']:
            penetrating_mask = state['cloth'].penetrating_mask
            triangles_penetrating = penetrating_mask[faces].unsqueeze(dim=0).contiguous()[0].any(dim=1)
        else:
            triangles_penetrating = None

        vertex_type = state['cloth'].vertex_type.squeeze()
        pinned_mask = vertex_type == 3

        riz_list = []

        iter = 0

        max_riz_size = 0
        while True:
            collisions_tri = find_close_faces(verts1, faces, threshold=self.mcfg.riz_epsilon)

            if triangles_penetrating is not None:
                collision_penetrating_mask = triangles_penetrating[collisions_tri[:, :2]].any(dim=1)[..., 0]
                collision_nonpenetrating_mask = torch.logical_not(collision_penetrating_mask)
                collisions_tri = collisions_tri[collision_nonpenetrating_mask]

            if collisions_tri.shape[0] == 0:
                break

            n_edges = 0 if edges is None else edges.shape[0]
            riz_list, edges = self.collision_helper.make_rigid_impact_zones_tritri(faces, collisions_tri,
                                                                                   edges=edges)

            riz_sizes = [x.shape[0] for x in riz_list]
            if len(riz_sizes) > 0:
                max_size = np.max(riz_sizes)
                max_riz_size = max(max_riz_size, max_size)
                if self.mcfg.max_riz_size > 0:
                    if max_size > self.mcfg.max_riz_size:
                        break

            if edges.shape[0] == n_edges:
                break

            iter += 1

            for riz in riz_list:

                rigid_dx = self.compute_riz_deltas(riz, pinned_mask, mass, verts0, dx)
                verts1[riz] = verts0[riz] + rigid_dx
                velocity_dx[riz] = rigid_dx

            if iter > self.mcfg.riz_max_steps_total:
                print(f' RIZ reached {self.mcfg.riz_max_steps_total}')
                print('edges', edges.shape[0])

        if metrics_dict is not None:
            label = label + 'riz_iters'
            metrics_dict[label].append(iter)
            label = label + 'max_riz_size'
            metrics_dict[label].append(max_riz_size)

        state['cloth'].pred_pos = verts1


        return state


    def impulses_compute_partial(self, state, metrics_dict=None, label=None):

        faces = self.get_faces(state)
        verts0 = state['cloth'].pos.clone()
        verts1 = state['cloth'].pred_pos.clone()
        mass = state['cloth'].v_mass.clone()

        # if self.mcfg.double_precision_impulse:
        #     verts0 = verts0.double()
        #     verts1 = verts1.double()
        #     mass = mass.double()

        if 'penetrating_mask' in state['cloth']:
            penetrating_mask = state['cloth'].penetrating_mask
            triangles_penetrating = penetrating_mask[faces].unsqueeze(dim=0).contiguous()
        else:
            triangles_penetrating = None

        vertex_type = state['cloth'].vertex_type.squeeze()
        pinned_mask = vertex_type == 3

        if self.mcfg.pinned_mass > 0:
            mass[pinned_mask] = self.mcfg.pinned_mass

        unpinned_mask = torch.logical_not(pinned_mask)
        unpinned_mask = unpinned_mask[:, None]

        vertex_dx_sum = torch.zeros_like(verts1)
        vertex_dv_sum = torch.zeros_like(verts1)
        triangles_mass = mass[faces].unsqueeze(dim=0).contiguous()

        ncoll = None

        iter = 0
        # print('\n')

        w = 1
        impulsed_points = []
        faces_to_check = None
        for i in range(self.mcfg.n_impulse_iters):
            # print(f"step {i}")
            verts1_curr = verts1 + vertex_dx_sum

            triangles1 = verts0[faces].unsqueeze(dim=0).contiguous()
            triangles2 = verts1_curr[faces].unsqueeze(dim=0).contiguous()

            bboxes, tree = cccollisions.bvh_motion(triangles1, triangles2)

            if faces_to_check is None:
                imp_dv, imp_dx, imp_counter = cccollisions.compute_impulses(bboxes, tree, triangles1, triangles2,
                                                                                triangles_mass,
                                                                                32 * 3, 16)
            else:
                imp_dv, imp_dx, imp_counter = cccollisions.compute_impulses_partial(bboxes, tree, triangles1,
                                                                                        triangles2,
                                                                                        triangles_mass, faces_to_check,
                                                                                        32 * 3, 16)

            imp_counter = imp_counter.long()


            if triangles_penetrating is not None:
                imp_counter = imp_counter * torch.logical_not(triangles_penetrating[..., 0])
                imp_dx = imp_dx * torch.logical_not(triangles_penetrating)
                imp_dv = imp_dv * torch.logical_not(triangles_penetrating)


            if ncoll is None:
                ncoll = imp_counter.sum().item() / 4

            if self.mcfg.max_ncoll > 0 and ncoll > self.mcfg.max_ncoll:

                break

            if imp_counter.sum() == 0:
                break

            vertex_dx_sum, vertex_dv_sum, faces_to_check = update_verts(vertex_dx_sum, vertex_dv_sum, verts1, faces,
                                                                        imp_counter, imp_dx, imp_dv, unpinned_mask, w=w)


            iter += 1

        if ncoll is None:
            ncoll = 0

        if metrics_dict is not None:
            label_iter = label + 'impulse_iters'
            metrics_dict[label_iter].append(iter)
            label_ncoll = label + 'impulse_stencil_ncoll'
            metrics_dict[label_ncoll].append(ncoll)
        return vertex_dx_sum, vertex_dv_sum



    def safecheck_impulses(self, state, metrics_dict=None, label='', update=True):
        vertex_dx_sum, vertex_dv_sum = self.impulses_compute_partial(state, metrics_dict, label)

        pred_pos = state['cloth'].pred_pos + vertex_dx_sum
        timestep = state['cloth'].timestep

        pred_velocity = state['cloth'].pred_velocity + vertex_dv_sum

        add_field_to_pyg_batch(state, 'hc_impulse_dx', vertex_dx_sum, 'cloth', 'pos')
        add_field_to_pyg_batch(state, 'hc_impulse_dv', vertex_dv_sum, 'cloth', 'pos')

        if update:
            state['cloth'].pred_pos = pred_pos
            state['cloth'].pred_velocity = pred_velocity
        else:
            add_field_to_pyg_batch(state, 'hc_impulse_pos', pred_pos, 'cloth', 'pos')
            add_field_to_pyg_batch(state, 'hc_impulse_velocity', pred_velocity, 'cloth', 'pos')

        return state



    @staticmethod
    def calc_tritri_collisions(sample, prev=False, threshold=0.):
        pos = sample['cloth'].pos if prev else sample['cloth'].pred_pos
        pos = pos.double()

        collisions_tri = find_close_faces(pos, sample['cloth'].faces_batch.T, threshold=threshold)

        if 'penetrating_mask' in sample['cloth']:
            faces = sample['cloth'].faces_batch.T

            penetrating_mask = sample['cloth'].penetrating_mask
            triangles_penetrating = penetrating_mask[faces].unsqueeze(dim=0).contiguous()[0].any(dim=1)

            collision_penetrating_mask = triangles_penetrating[collisions_tri[:, :2]].any(dim=1)[..., 0]
            collision_nonpenetrating_mask = torch.logical_not(collision_penetrating_mask)
            collisions_tri = collisions_tri[collision_nonpenetrating_mask]

        if 'faces_cutout_mask_batch' in sample['cloth']:
            faces_mask = sample['cloth'].faces_cutout_mask_batch[0]
            collision_mask = faces_mask[collisions_tri].all(dim=-1)
            collisions_tri = collisions_tri[collision_mask]

        return collisions_tri.shape[0]


    @staticmethod
    def calc_tritri_collisions2(sample, obj_key='cloth', verts_key='pred_pos', threshold=0.):
        pos = sample[obj_key][verts_key]
        faces = sample[obj_key].faces_batch.T
        pos = pos.double()

        collisions_tri = find_close_faces(pos, faces, threshold=threshold)

        if 'penetrating_mask' in sample[obj_key]:
            penetrating_mask = sample[obj_key].penetrating_mask
            triangles_penetrating = penetrating_mask[faces].unsqueeze(dim=0).contiguous()[0].any(dim=1)

            collision_penetrating_mask = triangles_penetrating[collisions_tri[:, :2]].any(dim=1)[..., 0]
            collision_nonpenetrating_mask = torch.logical_not(collision_penetrating_mask)
            collisions_tri = collisions_tri[collision_nonpenetrating_mask]

        if 'faces_cutout_mask_batch' in sample[obj_key]:
            faces_mask = sample[obj_key].faces_cutout_mask_batch[0]
            collision_mask = faces_mask[collisions_tri].all(dim=-1)
            collisions_tri = collisions_tri[collision_mask]

        return collisions_tri.shape[0]


    @staticmethod
    def mark_penetrating_faces(sample, threshold=0., object='cloth', use_target=False, dummy=False):

        if object=='obstacle' and 'obstacle' not in sample.node_types:
            return sample

        B = sample.num_graphs
        new_examples = []
        for i in range(B):
            example = sample.get_example(i)

            faces = example[object].faces_batch.T
            pos = example[object].pos

            if len(pos.shape) == 3:
                pos = pos[:, 0]

            if dummy:
                node_mask = torch.ones_like(pos[:, 0]).bool()
                faces_mask = torch.ones_like(faces[:, :1]).bool()
                example[object].cutout_mask = node_mask
                example[object].faces_cutout_mask_batch = faces_mask.T
                new_examples.append(example)
                # assert False, "dummy"
                continue


            collisions_tri = find_close_faces(pos, faces, threshold=threshold)
            unique_faces = torch.unique(collisions_tri[:, :2])
            faces_mask = torch.ones_like(faces[:, 0]).bool()
            faces_mask[unique_faces] = 0

            if use_target:
                target_pos = example[object].target_pos
                collisions_tri_next = find_close_faces(target_pos, faces, threshold=threshold)
                unique_faces_next = torch.unique(collisions_tri_next[:, :2])
                faces_mask[unique_faces_next] = 0

            faces_enabled = faces[faces_mask]
            node_ids = torch.unique(faces_enabled)
            node_mask = torch.zeros_like(pos[:, 0]).bool()
            node_mask[node_ids] = 1
            faces_mask = faces_mask[None]

            example[object].cutout_mask = node_mask
            example[object].faces_cutout_mask_batch = faces_mask

            new_examples.append(example)
        sample_new = Batch.from_data_list(new_examples)
        return sample_new



def update_verts(vertex_dx_sum, vertex_dv_sum, verts1, faces, imp_counter, imp_dx, imp_dv,unpinned_mask=None, w=1.):
    vertex_counts = torch.zeros_like(verts1[:, 0]).long()
    vertex_dx = torch.zeros_like(verts1)
    vertex_dv = torch.zeros_like(verts1)

    vertex_counts = torch_scatter.scatter(imp_counter.reshape(-1), faces.reshape(-1), dim=0, out=vertex_counts)
    vertex_changed = vertex_counts > 0
    faces_changed = vertex_changed[faces].any(dim=-1, keepdim=True)

    vertex_counts = vertex_counts[:, None]
    vertex_counts[vertex_counts == 0] = 1

    torch_scatter.scatter(imp_dx.reshape(-1, 3), faces.reshape(-1), dim=0, out=vertex_dx)
    torch_scatter.scatter(imp_dv.reshape(-1, 3), faces.reshape(-1), dim=0, out=vertex_dv)

    vertex_dx = vertex_dx / vertex_counts
    vertex_dv = vertex_dv / vertex_counts

    vertex_dx = vertex_dx * w
    vertex_dv = vertex_dv * w

    vertex_dx_sum = vertex_dx_sum + vertex_dx
    vertex_dv_sum = vertex_dv_sum + vertex_dv

    return vertex_dx_sum, vertex_dv_sum, faces_changed