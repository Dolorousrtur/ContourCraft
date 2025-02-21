from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from omegaconf import II
from torch import nn

from utils.cloth_and_material import FaceNormals
from utils.icontour import compute_igrad_loss


@dataclass
class Config:
    weight_start: float = 0.
    weight_max: float = 1e+4
    step_start: int = 30000
    step_max: int = 40000

    detach_coords: bool = False
    detach_coeffs: bool = False
    detach_aux_bary: bool = False
    detach_aux_edges: bool = False
    only_edgeloss: bool = False
    device: str = II('device')

    enable_attractions: bool = II("experiment.enable_attractions")


def create(mcfg):
    return Criterion(mcfg)


class Criterion(nn.Module):
    def __init__(self, mcfg):
        super().__init__()
        self.mcfg = mcfg
        self.name = 'icontour'

    def get_weight(self, iter):
        n_rampup_iterations = self.mcfg.step_max - self.mcfg.step_start
        iter = iter - self.mcfg.step_start
        iter = max(iter, 0)
        progress = iter / n_rampup_iterations
        progress = min(progress, 1.)
        weight = self.mcfg.weight_start + (self.mcfg.weight_max - self.mcfg.weight_start) * progress
        return weight


    def forward(self, sample):
        B = sample.num_graphs
        iter_num = sample['cloth'].iter[0].item()
        weight = self.get_weight(iter_num)
        device = sample['cloth'].pos.device

        n_attraction_edges = sample['cloth', 'attraction_edge', 'cloth'].edge_index.shape[1]

        enable_attractions = self.mcfg.enable_attractions

        if weight == 0 or not enable_attractions:
            return dict(loss=torch.FloatTensor([0]).to(device), weight=weight)

        if n_attraction_edges == 0:
            return dict(loss=torch.FloatTensor([0]).to(device), weight=weight)
        
        step = sample['cloth'].step[0].item()
        if step == 0:
            return dict(loss=torch.FloatTensor([0]).to(device), weight=weight)

        verts = sample['cloth'].pred_pos
        faces = sample['cloth'].faces_batch.T


        loss = compute_igrad_loss(verts, faces, detach_coords=self.mcfg.detach_coords, detach_coeffs=self.mcfg.detach_coeffs,
                                  detach_aux_bary=self.mcfg.detach_aux_bary, detach_aux_edges=self.mcfg.detach_aux_edges, 
                                  only_edgeloss=self.mcfg.only_edgeloss)
        
        if loss.grad_fn is None:
            grad = torch.zeros_like(verts)

        else:
            loss = loss * weight
            grad = torch.autograd.grad(loss, verts, retain_graph=True)[0]
            grad[grad != grad] = 0

        return dict(loss=loss.detach(), weight=weight, gradient=grad)