from dataclasses import dataclass

import torch
from omegaconf import II
from torch import nn

from utils.common import add_field_to_pyg_batch


@dataclass
class Config:
    weight: float = 1.


def create(mcfg, **kwargs):
    return Criterion(mcfg)


class Criterion(nn.Module):
    def __init__(self, mcfg):
        super().__init__()
        self.mcfg = mcfg
        self.name = 'tv'

    def forward(self, sample):
        cloth_sample = sample['cloth']

        lame_mu_input = cloth_sample.lame_mu_input
        lame_lambda_input = cloth_sample.lame_lambda_input
        bending_coeff_input = cloth_sample.bending_coeff_input

        edge_index = sample['cloth', 'mesh_edge', 'cloth'].edge_index.T


        losses = []

        for m in [lame_mu_input, lame_lambda_input, bending_coeff_input]:
            mvec_edge = m[edge_index][..., 0]
            tv_loss = (mvec_edge[:, 0] - mvec_edge[:, 1]).pow(2).sum(dim=-1).mean()
            losses.append(tv_loss)

        loss = torch.sum(torch.stack(losses))


        loss = loss * self.mcfg.weight


        return dict(loss=loss, weight=self.mcfg.weight)

