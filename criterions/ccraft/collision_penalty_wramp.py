from dataclasses import dataclass

import torch
from torch import nn

from utils.cloth_and_material import FaceNormals
from utils.common import gather
from utils.warp_u.proximity import get_closest_nodes_and_faces_pt_dummmy


@dataclass
class Config:
    weight_start: float = 1e+3
    weight_max: float = 1e+5
    start_rampup_iteration: int = 50000
    n_rampup_iterations: int = 100000
    eps: float = 1e-3
    correspondence_eps: float = 3e-2


def create(mcfg):
    return Criterion(mcfg, weight_start=mcfg.weight_start, weight_max=mcfg.weight_max,
                     start_rampup_iteration=mcfg.start_rampup_iteration, n_rampup_iterations=mcfg.n_rampup_iterations,
                     eps=mcfg.eps)


class Criterion(nn.Module):
    def __init__(self, mcfg, weight_start, weight_max, start_rampup_iteration, n_rampup_iterations, eps=1e-3):
        super().__init__()
        self.mcfg=mcfg
        self.weight_start = weight_start
        self.weight_max = weight_max
        self.start_rampup_iteration = start_rampup_iteration
        self.n_rampup_iterations = n_rampup_iterations
        self.eps = eps
        self.f_normals_f = FaceNormals()
        self.name = 'collision_penalty'

    def get_weight(self, iter):
        iter = iter - self.start_rampup_iteration
        iter = max(iter, 0)
        progress = iter / self.n_rampup_iterations
        progress = min(progress, 1.)
        weight = self.weight_start + (self.weight_max - self.weight_start) * progress
        return weight

    def calc_loss(self, example):
        obstacle_next_pos = example['obstacle'].target_pos
        obstacle_curr_pos = example['obstacle'].pos
        obstacle_faces = example['obstacle'].faces_batch.T
        device = obstacle_next_pos.device

        curr_pos = example['cloth'].pos
        next_pos = example['cloth'].pred_pos

        if 'cutout_mask' in example['cloth']:
            node_mask = example['cloth'].cutout_mask
            curr_pos = curr_pos[node_mask]
            next_pos = next_pos[node_mask]

        if 'faces_cutout_mask_batch' in example['obstacle']:
            faces_mask = example['obstacle'].faces_cutout_mask_batch[0]
            obstacle_faces = obstacle_faces[faces_mask]


        # Find correspondences in current step
        indices_from, _, face_indices_to, _ = get_closest_nodes_and_faces_pt_dummmy(curr_pos, obstacle_curr_pos,
                                                                               obstacle_faces,
                                                                               self.mcfg.correspondence_eps)
        indices_from = indices_from.unsqueeze(-1).long()
        face_indices_to = face_indices_to.unsqueeze(-1).long()
        # Compute distances in the new step
        obstacle_face_next_pos = gather(obstacle_next_pos, obstacle_faces, 0, 1, 1).mean(dim=-2)
        obstacle_fn = self.f_normals_f(obstacle_next_pos.unsqueeze(0), obstacle_faces.unsqueeze(0))[0]

        nn_points = gather(obstacle_face_next_pos, face_indices_to, 0, 1, 1)
        nn_normals = gather(obstacle_fn, face_indices_to, 0, 1, 1)
        next_pos_from = next_pos[indices_from[:, 0]]

        nn_points = nn_points[:, 0]
        nn_normals = nn_normals[:, 0]
        device = next_pos.device

        distance = ((next_pos_from - nn_points) * nn_normals).sum(dim=-1)
        interpenetration = torch.maximum(self.eps - distance, torch.FloatTensor([0]).to(device))

        interpenetration = interpenetration.pow(3)
        loss = interpenetration.sum(-1)

        # return torch.FloatTensor([0]).to(device)
        return loss

    def forward(self, sample):
        B = sample.num_graphs
        if 'obstacle' not in sample.node_types:
            return dict()

        iter_num = sample['cloth'].iter[0].item()
        weight = self.get_weight(iter_num)

        loss_list = []
        for i in range(B):
            loss_list.append(
                self.calc_loss(sample.get_example(i)))

        loss = sum(loss_list) / B * weight

        return dict(loss=loss, weight=weight)
