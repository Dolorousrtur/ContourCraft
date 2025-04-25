
from dataclasses import dataclass

import torch

from material.utils import get_vertex_mass
from torch_geometric.data import Batch
from utils.common import gather, relative_between_log


@dataclass
class Config:
    density_min: float = 0.20022
    density_max: float = 0.20022
    lame_mu_min: float = 23600.0
    lame_mu_max: float = 23600.0
    lame_lambda_min: float = 44400
    lame_lambda_max: float = 44400
    bending_coeff_min: float = 3.9625778333333325e-05
    bending_coeff_max: float = 3.9625778333333325e-05

    density_init: float = 0.20022
    lame_mu_init: float = 23600.0
    lame_lambda_init: float = 44400
    bending_coeff_init: float = 3.9625778333333325e-05

class Material(torch.nn.Module):
    def __init__(self, mcfg, sample=None, state_dict=None):
        super().__init__()

        assert sample is not None or state_dict is not None

        if sample is not None:
            self.init_from_sample(mcfg, sample)
        else:
            self.init_from_state_dict(state_dict)

    def init_from_state_dict(self, state_dict):
        self.lame_mu_input = torch.nn.Parameter(state_dict['lame_mu_input'])
        self.lame_lambda_input = torch.nn.Parameter(state_dict['lame_lambda_input'])
        self.bending_coeff_input = torch.nn.Parameter(state_dict['bending_coeff_input'])
        self.v_mass = torch.nn.Parameter(state_dict['v_mass'])

        self.rest_mults = torch.nn.ParameterDict()
        for k in state_dict.keys():
            if not k.startswith('rest_mults'):
                continue
            
            edge_key = k.split('.')[1]
            self.rest_mults[edge_key] = torch.nn.Parameter(state_dict[k])

    def init_from_sample(self, mcfg, sample):


        lame_mu_init_norm = relative_between_log(mcfg.lame_mu_min, mcfg.lame_mu_max, mcfg.lame_mu_init)
        lame_lambda_init_norm = relative_between_log(mcfg.lame_lambda_min, mcfg.lame_lambda_max, mcfg.lame_lambda_init)
        bending_coeff_init_norm = relative_between_log(mcfg.bending_coeff_min, mcfg.bending_coeff_max, mcfg.bending_coeff_init)


        pos = sample['cloth'].rest_pos
        faces = sample['cloth'].faces_batch.T
        n_verts = sample['cloth'].pos.shape[0]
        density = torch.tensor(mcfg.density_init, dtype=torch.float32)

        device = pos.device


        lame_mu_input = torch.tensor(lame_mu_init_norm, dtype=torch.float32).expand(n_verts)[:, None]
        self.lame_mu_input = torch.nn.Parameter(lame_mu_input).to(device)

        lame_lambda_input = torch.tensor(lame_lambda_init_norm, dtype=torch.float32).expand(n_verts)[:, None]
        self.lame_lambda_input = torch.nn.Parameter(lame_lambda_input).to(device)

        bending_coeff_input = torch.tensor(bending_coeff_init_norm, dtype=torch.float32).expand(n_verts)[:, None]
        self.bending_coeff_input = torch.nn.Parameter(bending_coeff_input).to(device)


        self.rest_mults = torch.nn.ParameterDict()

        v_mass = get_vertex_mass(pos, faces, density)[:, None]
        self.v_mass = torch.nn.Parameter(v_mass).to(device)
        # print('v_mass', self.v_mass.device)

        self.collect_rest_edges(sample.get_example(0))



    def get_relative_pos(self, pos, edges):
        edges_pos = gather(pos, edges, 0, 1, 1).permute(0, 2, 1)
        pos_senders, pos_receivers = edges_pos.unbind(-1)
        relative_pos = pos_senders - pos_receivers
        return relative_pos
    
    def collect_rest_edges(self, example):
        rest_pos = example['cloth'].rest_pos
        for _, edge_label, _ in example.edge_types:
            if edge_label.startswith('coarse') or edge_label.startswith('mesh'):
                edge_index = example['cloth', edge_label, 'cloth'].edge_index.T

                relative_pos = self.get_relative_pos(rest_pos, edge_index)

                rest_mult = torch.ones(relative_pos.shape[0], 1).to(relative_pos.device)

                self.rest_mults[edge_label] = torch.nn.Parameter(rest_mult, requires_grad=True)


    def _apply_nodes(self, example):
        example['cloth'].lame_mu_input = self.lame_mu_input
        example['cloth'].lame_lambda_input = self.lame_lambda_input
        example['cloth'].bending_coeff_input = self.bending_coeff_input

        example['cloth'].v_mass = self.v_mass

        if 'obstacle' in example.node_types:
            pos = example['obstacle'].pos
            ones = torch.ones(pos.shape[0], 1).to(pos.device)
            example['obstacle'].lame_mu_input = ones.detach().clone()
            example['obstacle'].lame_lambda_input = ones.detach().clone()
            example['obstacle'].bending_coeff_input = ones.detach().clone()

        return example
    
    def _apply_edges(self, example, edge_label):
        edge_index = example['cloth', edge_label, 'cloth'].edge_index.T

        for m in ['lame_mu_input', 'lame_lambda_input', 'bending_coeff_input']:
            mvec = example['cloth'][m]

            mvec_edge = mvec[edge_index].mean(-2)

            example['cloth', edge_label, 'cloth'][m] = mvec_edge


        if edge_label in self.rest_mults:
            rest_mult = self.rest_mults[edge_label]
            rest_pos = example['cloth'].rest_pos
            relative_rest_pos = self.get_relative_pos(rest_pos, edge_index)

            example['cloth', edge_label, 'cloth'].relative_rest_pos = relative_rest_pos * rest_mult

        return example
    
    def get_material(self):
        out_dict = {
            'lame_mu_input': self.lame_mu_input,
            'lame_lambda_input': self.lame_lambda_input,
            'bending_coeff_input': self.bending_coeff_input,
            'v_mass': self.v_mass
        }

        out_dict['rest_mults'] = {}
        for k, v in self.rest_mults.items():
            out_dict[f'rest_mults'][k] = v

        return out_dict

    def apply_material(self, sample):
        n_graphs = len(sample)


        examples = []
        for i in range(n_graphs):
            example = sample.get_example(i)


            example = self._apply_nodes(example)
            example = self._apply_edges(example, 'mesh_edge')


            for _, edge_label, _ in example.edge_types:
                if edge_label.startswith('coarse') or 'repulsion' in edge_label or 'attraction' in edge_label:
                    example = self._apply_edges(example, edge_label)

            examples.append(example)

        sample = Batch.from_data_list(examples)

        return sample




    