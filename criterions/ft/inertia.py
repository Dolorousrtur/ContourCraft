from dataclasses import dataclass

import torch
from omegaconf import II
from torch import nn

from utils.cloth_and_material import get_vertex_mass
from utils.common import add_field_to_pyg_batch


@dataclass
class Config:
    weight: float = 1.
    density: float = 0.20022


def create(mcfg, **kwargs):
    return Criterion(mcfg)


class Criterion(nn.Module):
    def __init__(self, mcfg):
        super().__init__()
        self.mcfg = mcfg
        self.name = 'inertia'

    def forward(self, sample):
        cloth_sample = sample['cloth']
        timestep = cloth_sample.timestep

        pred_pos = cloth_sample.pred_pos
        pos = cloth_sample.pos
        velocity = cloth_sample.velocity
        ts = timestep[0]
        velocity_dx = velocity
        device = pos.device



        B = sample.num_graphs

        x_hat = pos + velocity_dx
        x_diff = pred_pos - x_hat


        rest_pos = cloth_sample.rest_pos
        faces = cloth_sample.faces_batch.T
        mass = get_vertex_mass(rest_pos, faces, self.mcfg.density)
        mass = mass.unsqueeze(-1).to(device)



        num = (x_diff * mass * x_diff).sum(dim=-1).unsqueeze(1)

        den = 2 * ts ** 2

        loss = num / den


        if 'cutout_mask' in sample['cloth']:
            node_mask = cloth_sample.cutout_mask
            loss = loss[node_mask]

        loss = loss.sum() / B * self.mcfg.weight

        return dict(loss=loss, weight=self.mcfg.weight)
