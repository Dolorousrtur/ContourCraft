from dataclasses import dataclass

from torch import nn


@dataclass
class Config:
    weight: float = 1.
    g: float = 9.81
    z_axis: int = 2


def create(mcfg):
    return Criterion(weight=mcfg.weight, z_axis=mcfg.z_axis)


class Criterion(nn.Module):
    def __init__(self, weight, g=9.81, z_axis=2):
        super().__init__()
        self.weight = weight
        self.g = g
        self.z_axis = z_axis

        self.name = 'gravitational_energy'

    def forward(self, sample):
        cloth_sample = sample['cloth']
        pred_pos = cloth_sample.pred_pos


        B = sample.num_graphs
        v_mass = cloth_sample.v_mass
        U = self.g * v_mass[..., 0] * pred_pos[..., self.z_axis]

        if 'cutout_mask' in sample['cloth']:
            node_mask = cloth_sample.cutout_mask
            U = U[node_mask]

        loss = U.sum() / B * self.weight

        return dict(loss=loss, weight=self.weight)
