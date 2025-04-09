from dataclasses import dataclass

import torch
from omegaconf import II
from torch import nn

from utils.common import add_field_to_pyg_batch


@dataclass
class Config:
    weight: float = 1.
    ts_agnostic: bool = II("ts_agnostic")
    initial_ts: float = II("experiment.initial_ts")


def create(mcfg):
    return Criterion(mcfg)


class Criterion(nn.Module):
    def __init__(self, mcfg):
        super().__init__()
        self.mcfg = mcfg
        self.name = 'inertia'

    def forward(self, sample):
        cloth_sample = sample['cloth']
        timestep = cloth_sample.timestep
        is_init = cloth_sample.is_init

        pred_pos = cloth_sample.pred_pos
        pos = cloth_sample.pos
        velocity = cloth_sample.velocity

        if is_init:
            ts = self.mcfg.initial_ts
        else:
            ts = timestep[0]

        if self.mcfg.ts_agnostic:
            velocity_dx = velocity * timestep[0]
        else:
            velocity_dx = velocity


        mass = cloth_sample.v_mass

        B = sample.num_graphs

        x_hat = pos + velocity_dx
        x_diff = pred_pos - x_hat


        num = (x_diff * mass * x_diff).sum(dim=-1).unsqueeze(1)

        den = 2 * ts ** 2

        loss = num / den


        if 'cutout_mask' in sample['cloth']:
            node_mask = cloth_sample.cutout_mask
            loss = loss[node_mask]

        loss = loss.sum() / B * self.mcfg.weight

        return dict(loss=loss, weight=self.mcfg.weight)
