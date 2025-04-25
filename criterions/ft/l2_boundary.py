from dataclasses import dataclass

import torch
from omegaconf import II
from torch import nn

from utils.cloth_and_material import FaceNormals
from utils.common import add_field_to_pyg_batch
from pytorch3d.ops import knn_points
from utils.common import gather



@dataclass
class Config:
    weight: float = 1.
    omit_penetrations: bool = False
    eps: float = 3e-3


def create(mcfg, **kwargs):
    return Criterion(mcfg)


class Criterion(nn.Module):
    def __init__(self, mcfg):
        super().__init__()
        self.mcfg = mcfg
        self.name = 'l2_boundary'
        self.normals_f = FaceNormals()

    def make_penetration_mask(self, sample):
        target_pos = sample['cloth'].target_pos.unsqueeze(0)
        obstacle_pos = sample['obstacle'].target_pos.unsqueeze(0)
        obstacle_faces = sample['obstacle'].faces_batch.T.unsqueeze(0)

        obstacle_face_pos = gather(obstacle_pos, obstacle_faces, 1, 2, 2).mean(dim=2)
        obstacle_fn = self.normals_f(obstacle_pos, obstacle_faces)
        _, nn_idx, nn_points = knn_points(target_pos, obstacle_face_pos, return_nn=True)
        nn_points = nn_points[:, :, 0]

        nn_normals = gather(obstacle_fn, nn_idx, 1, 2, 2)
        nn_normals = nn_normals[:, :, 0]

        direction = target_pos - nn_points
        distance = (direction * nn_normals).sum(dim=-1)
        distance = distance[0]

        # penetration_mask = distance > 0
        penetration_mask = distance > self.mcfg.eps

        return penetration_mask



    def forward(self, sample):
        cloth_sample = sample['cloth']
        pred_pos = cloth_sample.pred_pos
        target_pos = cloth_sample.target_pos

        boundary_mask = cloth_sample.boundary_mask
        loss = (pred_pos - target_pos).pow(2)
        loss = loss * boundary_mask.unsqueeze(-1)
        loss = loss.sum(dim=-1)

        if self.mcfg.omit_penetrations:
            penetration_mask = self.make_penetration_mask(sample)
            loss = loss * penetration_mask.float()




        loss = loss.mean() * self.mcfg.weight


        return dict(loss=loss, weight=self.mcfg.weight)

