from dataclasses import dataclass

import einops
import torch
from torch import nn

from utils.cloth_and_material import edges_3d_to_2d, gather_triangles, get_shape_matrix
from utils.common import make_pervertex_tensor_from_lens


@dataclass
class Config:
    weight: float = 1.
    thickness: float = 4.7e-4
    lame_mu: float = 63636
    lame_lambda: float = 93333


def create(mcfg, **kwargs):
    return Criterion(mcfg, weight=mcfg.weight)


def deformation_gradient(triangles, Dm_inv):
    Ds = get_shape_matrix(triangles)

    return Ds @ Dm_inv


def green_strain_tensor(F):
    device = F.device
    I = torch.eye(2, dtype=F.dtype).to(device)

    Ft = F.permute(0, 2, 1)
    return 0.5 * (Ft @ F - I)


def make_Dm_inv(v, f):
    tri_m = gather_triangles(v.unsqueeze(0), f)[0]

    edges = get_shape_matrix(tri_m)
    edges = edges.permute(0, 2, 1)
    edges_2d = edges_3d_to_2d(edges).permute(0, 2, 1)
    Dm_inv = torch.inverse(edges_2d)
    return Dm_inv

class Criterion(nn.Module):
    def __init__(self, mcfg, weight):
        super().__init__()
        self.weight = weight
        self.thickness = mcfg.thickness
        self.lame_mu = mcfg.lame_mu
        self.lame_lambda = mcfg.lame_lambda
        self.name = 'stretching'

    def create_stack(self, triangles_list, param):
        lens = [x.shape[0] for x in triangles_list]
        stack = make_pervertex_tensor_from_lens(lens, param)[:, 0]
        return stack


    def forward(self, sample):

        target_pos = sample['cloth'].target_pos
        faces = sample['cloth'].faces_batch.T
        Dm_inv = make_Dm_inv(target_pos, faces)


        f_area = sample['cloth'].f_area[None, ..., 0]
        device = Dm_inv.device

        B = sample.num_graphs

        triangles_list = []
        for i in range(B):
            example = sample.get_example(i)
            v = example['cloth'].pred_pos
            f = example['cloth'].faces_batch.T

            triangles = gather_triangles(v.unsqueeze(0), f)[0]
            triangles_list.append(triangles)

        triangles = torch.cat(triangles_list, dim=0)

        F = deformation_gradient(triangles, Dm_inv)
        G = green_strain_tensor(F)

        I = torch.eye(2).to(device)
        I = einops.repeat(I, 'm n -> k m n', k=G.shape[0])

        G_trace = G.diagonal(dim1=-1, dim2=-2).sum(-1)  # trace

        S = self.lame_mu * G + 0.5 * self.lame_lambda * G_trace[:, None, None] * I
        energy_density_matrix = S.permute(0, 2, 1) @ G
        energy_density = energy_density_matrix.diagonal(dim1=-1, dim2=-2).sum(-1)  # trace
        f_area = f_area[0]

        energy = f_area * self.thickness * energy_density

        energy[energy != energy] = 0   # remove NaNs

        loss = energy.sum() / B * self.weight

        return dict(loss=loss, weight=self.weight)
