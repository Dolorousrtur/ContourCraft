from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import II
from torch import nn

from utils.cloth_and_material import FaceNormals
from utils.common import NodeType
from utils.defaults import DEFAULTS
from utils.selfcollisions import get_node2face_signed_distance, get_static_proximity, \
    CollisionHelper
from utils.warp_u.proximity import get_proximity_self_pt_dummmy


@dataclass
class Config:
    weight_start: float = 0.
    weight_max: float = 1e+4
    step_start: int = 30000
    step_max: int = 40000

    pinned_relative_weight: float = 1e+2
    penalty_eps: float = 1e-3
    correspondence_eps: float = 5e-3
    detach_faces: bool = False
    device: str = II('device')

    enable_attractions: bool = II("experiment.enable_attractions")


def create(mcfg):
    return Criterion(mcfg)


class Criterion(nn.Module):
    def __init__(self, mcfg):
        super().__init__()
        self.mcfg = mcfg
        self.pinned_relative_weight = mcfg.pinned_relative_weight
        self.correspondence_eps = mcfg.correspondence_eps
        self.penalty_eps = mcfg.penalty_eps
        self.name = 'repulsion'
        self.face_normals_f = FaceNormals()
        self.collision_helper = CollisionHelper(self.mcfg.device)

    def get_weight(self, iter):
        n_rampup_iterations = self.mcfg.step_max - self.mcfg.step_start
        iter = iter - self.mcfg.step_start
        iter = max(iter, 0)
        progress = iter / n_rampup_iterations
        progress = min(progress, 1.)
        weight = self.mcfg.weight_start + (self.mcfg.weight_max - self.mcfg.weight_start) * progress
        return weight

    def calc_loss(self, example, edge_key, sign=1):
        prev_pos = example['cloth'].pos
        pred_pos = example['cloth'].pred_pos
        device = pred_pos.device
        cloth_faces = example['cloth'].faces_batch.T
        vertex_type = example['cloth'].vertex_type

        step = example['cloth'].step[0].item()

        if step == 0:
            return torch.FloatTensor([0.]).to(prev_pos.device)

        if 'faces_to' not in example['cloth', edge_key, 'cloth']:
            return torch.FloatTensor([0]).to(device)

        pinned_mask = vertex_type == NodeType.HANDLE

        triangles_pinned = pinned_mask[cloth_faces].unsqueeze(dim=0).contiguous()[0].any(dim=1)


        faces_to = example['cloth', edge_key, 'cloth'].faces_to[:, 0]
        nodes_from = example['cloth', edge_key, 'cloth'].nodes_from[:, 0]

        dists_normal_prev = get_node2face_signed_distance(prev_pos, cloth_faces, self.face_normals_f, nodes_from,
                                                          faces_to, detach_faces=self.mcfg.detach_faces)
        stashed_mask = dists_normal_prev.abs() < self.mcfg.correspondence_eps

        dists_normal_prev = dists_normal_prev[stashed_mask]
        nodes_from = nodes_from[stashed_mask]
        faces_to = faces_to[stashed_mask]

        points_pinned_mask = pinned_mask[nodes_from][:, 0]
        faces_pinned_mask = triangles_pinned[faces_to][:, 0]
        full_pinned_mask = torch.logical_or(points_pinned_mask, faces_pinned_mask)

        if nodes_from.shape[0] == 0:
            return torch.FloatTensor([0]).to(device)

        dists_normal_curr = get_node2face_signed_distance(pred_pos, cloth_faces, self.face_normals_f, nodes_from,
                                                          faces_to, detach_faces=self.mcfg.detach_faces)

        prev_sign = torch.sign(dists_normal_prev)
        dists_normal_curr *= prev_sign
        dists_normal_curr *= sign


        interpenetration = torch.maximum(self.penalty_eps - dists_normal_curr, torch.FloatTensor([0]).to(device))
        interpenetration[full_pinned_mask] = interpenetration[full_pinned_mask] * self.pinned_relative_weight

        loss = interpenetration.pow(3)
        loss = loss.sum(-1)

        return loss

    def forward(self, sample):
        B = sample.num_graphs
        iter_num = sample['cloth'].iter[0].item()
        weight = self.get_weight(iter_num)

        if weight == 0:
            return dict(loss=torch.FloatTensor([0]).to(sample['cloth'].pos.device), weight=weight)

        loss_list = []
        for i in range(B):
            repulsion_loss = self.calc_loss(sample.get_example(i), 'repulsion_edge', sign=1)
            loss_list.append(repulsion_loss)

        loss = sum(loss_list) / B * weight

        return dict(loss=loss, weight=weight)