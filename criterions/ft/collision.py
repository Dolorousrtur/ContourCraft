from dataclasses import dataclass

import torch
from pytorch3d.ops import knn_points
from torch import nn

from utils.cloth_and_material import FaceNormals
from utils.common import gather
from utils.warp_u.proximity import get_closest_nodes_and_faces_pt_dummmy


@dataclass
class Config:
    weight: float = 1e+3
    eps: float = 1e-3
    correspondence_eps: float = 3e-2


def create(mcfg, **kwargs):
    return Criterion(mcfg)


class Criterion(nn.Module):
    def __init__(self, mcfg):
        super().__init__()
        self.mcfg=mcfg
        self.weight = mcfg.weight
        self.eps = mcfg.eps
        self.f_normals_f = FaceNormals()
        self.name = 'collision_penalty'

    def calc_loss(self, example):
        obstacle_next_pos = example['obstacle'].target_pos
        obstacle_curr_pos = example['obstacle'].pos
        obstacle_faces = example['obstacle'].faces_batch.T
        device = obstacle_next_pos.device

        curr_pos = example['cloth'].pos
        next_pos = example['cloth'].pred_pos


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

        weight = self.weight

        loss_list = []
        for i in range(B):
            loss_list.append(
                self.calc_loss(sample.get_example(i)))

        loss = sum(loss_list) / B * weight

        return dict(loss=loss, weight=weight)
