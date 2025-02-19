from dataclasses import dataclass

import torch
from torch import nn

from utils.cloth_and_material import FaceNormals
from utils.common import gather
import pytorch3d


@dataclass
class Config:
    weight: float = 1.


def create(mcfg):
    return Criterion(weight=mcfg.weight)


def make_rotation_matrix(n0, e):
    """
    n0: (N, 3)
    e: (N, 3)

    
    n0 ane e are orthogonal vectors
    create rotation matrices [Nx3x3] that 
    rotate n0 to be aligned with z axis (0,1,0)
    and e to be aligned with y axis (0,0,1)

    returns: (N, 3, 3)
    """
    
    n0 = n0 / torch.norm(n0, dim=-1, keepdim=True)
    e = e / torch.norm(e, dim=-1, keepdim=True)


    z_ours = n0
    y_ours = e
    x_ours = torch.cross(y_ours, z_ours)
    x_ours = x_ours / torch.norm(x_ours, dim=-1, keepdim=True)

    our_basis = torch.stack([x_ours, y_ours, z_ours], dim=-1)

    our_basis_quat = pytorch3d.transforms.matrix_to_quaternion(our_basis)
    our_basis_quat_inverse = pytorch3d.transforms.quaternion_invert(our_basis_quat)
    rot_mat = pytorch3d.transforms.quaternion_to_matrix(our_basis_quat_inverse)
    return rot_mat    

def rotate_pairs(n0, n1, e, rotmat):
    """
    n0: (N, 3)
    n1: (N, 3)
    e: (N, 3)

    rotmat: (N, 3, 3)

    returns: (N, 3), (N, 3), (N, 3)
    """

    n0 = torch.bmm(rotmat, n0.unsqueeze(-1)).squeeze(-1)
    n1 = torch.bmm(rotmat, n1.unsqueeze(-1)).squeeze(-1)
    e = torch.bmm(rotmat, e.unsqueeze(-1)).squeeze(-1)

    return n0, n1, e

class Criterion(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
        self.face_normals_f = FaceNormals()
        self.name = 'bending_energy'

        self.eps = 1e-6

    def calc_single(self, example):
        pred_pos = example['cloth'].pred_pos
        rest_pos = example['cloth'].rest_pos
        faces = example['cloth'].faces_batch.T
        f_connectivity = example['cloth'].f_connectivity
        f_connectivity_edges = example['cloth'].f_connectivity_edges
        f_area = example['cloth'].f_area
        bending_coeff = example['cloth'].bending_coeff

        # print('bending_coeff', bending_coeff.item())

        if 'faces_cutout_mask_batch' in example['cloth']:
            faces_mask = example['cloth'].faces_cutout_mask_batch[0]
            f_connectivity_mask = faces_mask[f_connectivity].all(dim=1)
            f_connectivity = f_connectivity[f_connectivity_mask]
            f_connectivity_edges = f_connectivity_edges[f_connectivity_mask]

        fn = self.face_normals_f(pred_pos.unsqueeze(0), faces.unsqueeze(0))[0]

        n = gather(fn, f_connectivity, 0, 1, 1)
        n0, n1 = torch.unbind(n, dim=-2)
        v = gather(pred_pos, f_connectivity_edges, 0, 1, 1)
        v0, v1 = torch.unbind(v, dim=-2)
        e = v1 - v0

        rot_mat = make_rotation_matrix(n0.detach(), e.detach())
        n0, n1, e = rotate_pairs(n0, n1, e, rot_mat)


        l = torch.norm(e, dim=-1, keepdim=True)
        e_norm = e / l

        fn_rest = self.face_normals_f(rest_pos.unsqueeze(0), faces.unsqueeze(0))[0]
        n_rest = gather(fn_rest, f_connectivity, 0, 1, 1)
        n0_rest, n1_rest = torch.unbind(n_rest, dim=-2)

        v_rest = gather(rest_pos, f_connectivity_edges, 0, 1, 1)
        v0_rest, v1_rest = torch.unbind(v_rest, dim=-2)
        e_rest = v1_rest - v0_rest

        rot_mat_rest = make_rotation_matrix(n0_rest.detach(), e_rest.detach())
        n0_rest, n1_rest, e_rest = rotate_pairs(n0_rest, n1_rest, e_rest, rot_mat_rest)

        f_area_repeat = f_area.repeat(1, f_connectivity.shape[-1])
        a = torch.gather(f_area_repeat, 0, f_connectivity).sum(dim=-1)

        cos = (n1 * n1_rest).sum(dim=-1)
        sin = (e_norm * torch.linalg.cross(n1, n1_rest)).sum(dim=-1)
        theta = torch.atan2(sin, cos)

        scale = l[..., 0] ** 2 / (4 * a)

        energy = bending_coeff * scale * (theta ** 2) / 2
        loss = energy.sum()
        return loss

    def forward(self, sample):

        loss_list = []
        B = sample.num_graphs
        for i in range(B):
            loss_sample = self.calc_single(sample.get_example(i))
            loss_list.append(loss_sample)

        loss = sum(loss_list) / B * self.weight
        # assert False

        return dict(loss=loss, weight=self.weight)
