from collections import defaultdict
from dataclasses import dataclass
import torch

from utils.arguments import load_module
from torch_geometric.data import Batch

from utils.common import gather


@dataclass
class OptimConfig:
    lr: float = 1e-4


@dataclass
class Config:
    dummy: float = 0.20022
    transfer_missing: bool = False
    use_inverse_scale: bool = False
    optimizer: OptimConfig = OptimConfig()

def create(module_config, **kwargs):
    return MaterialStack(module_config)

class MaterialStack(torch.nn.Module):
    def __init__(self, mcfg) -> None:
        super().__init__()
        self.mcfg = mcfg
        self.materials = torch.nn.ModuleDict()

        material_module = load_module('material', mcfg.material)
        self.material_class = material_module.Material

        material_name = list(mcfg.material.keys())[0]
        self.material_config = mcfg.material[material_name]

    def initialize(self, dataloader):
        tuple_set = set()
        for i, sample in enumerate(dataloader):
            garment_name = sample['garment_name'][0]
            sequence_name = sample['sequence_name'][0]
            if (garment_name, sequence_name) in tuple_set:
                break
            tuple_set.add((garment_name, sequence_name))
            self.materials[garment_name] = self.material_class(self.material_config, sample)

    def _stack_material_dicts(self, material_dicts):
        stacked_material_dict = defaultdict(list)
        for material_dict in material_dicts:
            for k, val in material_dict.items():
                if type(val) == dict:
                    if k not in stacked_material_dict:
                        stacked_material_dict[k] = {}

                    for kk, vv in val.items():
                        if kk not in stacked_material_dict[k]:
                            stacked_material_dict[k][kk] = []
                        stacked_material_dict[k][kk].append(vv)
                else:
                    stacked_material_dict[k].append(val)


        for k, v in stacked_material_dict.items():
            if type(v) == dict:
                # print('kk', v.keys())
                for kk, vv in v.items():
                    stacked_material_dict[k][kk] = torch.cat(vv, dim=0)
            else:
                stacked_material_dict[k] = torch.cat(v, dim=0)


        stacked_material_dict = dict(stacked_material_dict)
        return stacked_material_dict
    


    def _apply_nodes_multigarment(self, example, material_dict):
        example['cloth'].lame_mu_input = material_dict['lame_mu_input']
        example['cloth'].lame_lambda_input = material_dict['lame_lambda_input']
        example['cloth'].bending_coeff_input = material_dict['bending_coeff_input']
        example['cloth'].v_mass = material_dict['v_mass']


        if 'obstacle' in example.node_types:
            pos = example['obstacle'].pos
            ones = -torch.ones(pos.shape[0], 1).to(pos.device)
            example['obstacle'].lame_mu_input = ones.detach().clone()
            example['obstacle'].lame_lambda_input = ones.detach().clone()
            example['obstacle'].bending_coeff_input = ones.detach().clone()

        return example

    def get_relative_pos(self, pos, edges):
        edges_pos = gather(pos, edges, 0, 1, 1).permute(0, 2, 1)
        pos_senders, pos_receivers = edges_pos.unbind(-1)
        relative_pos = pos_senders - pos_receivers
        return relative_pos


    def _apply_edges_multigarment(self, example, edge_label, material_dict):
        edge_index = example['cloth', edge_label, 'cloth'].edge_index.T
        pos = example['cloth'].pos
        

        for m in ['lame_mu_input', 'lame_lambda_input', 'bending_coeff_input']:
            mvec = example['cloth'][m]

            mvec_edge = mvec[edge_index].mean(-2)

            example['cloth', edge_label, 'cloth'][m] = mvec_edge

        if 'rest_edges' in material_dict:
            if edge_label in material_dict['rest_edges']:
                # if 'mesh' in edge_label:
                #     continue
                # print('edge_label', edge_label)
                relative_rest_pos = material_dict['rest_edges'][edge_label]
                edge_index = example['cloth', edge_label, 'cloth'].edge_index
                example['cloth', edge_label, 'cloth'].relative_rest_pos = relative_rest_pos
                # inverse_scale = example['cloth', edge_label, 'cloth'].inverse_scale
        elif 'rest_mults'  in material_dict:
            if edge_label in material_dict['rest_mults']:
                rest_mult = material_dict['rest_mults'][edge_label]
                rest_pos = example['cloth'].rest_pos
                edge_index = example['cloth', edge_label, 'cloth'].edge_index.T
                relative_rest_pos = self.get_relative_pos(rest_pos, edge_index)


                relative_rest_pos = relative_rest_pos * rest_mult


                if self.mcfg.use_inverse_scale:
                    inverse_scale = example['cloth', edge_label, 'cloth'].inverse_scale
                    relative_rest_pos = relative_rest_pos * inverse_scale

                example['cloth', edge_label, 'cloth'].relative_rest_pos = relative_rest_pos
        else:
            raise ValueError('No rest_edges or rest_mults in material_dict')

        return example


    def _apply_material_multigarment(self, sample):
        gname = sample['garment_name'][0]
        garments = gname.split(',')
        garments = [g.split('::')[-1] for g in garments]

        material_dicts = []


        all_gnames = list(self.materials.keys())

        for i, garment_name in enumerate(garments):
            if garment_name in all_gnames:
                mdict = self.materials[garment_name].get_material()
            elif self.mcfg.transfer_missing:
                print(f'[WARNING] Missing material for {garment_name}, transferring from {all_gnames[0]}')
                mdict = self.materials[all_gnames[0]].transfer_material(sample, i)

            material_dicts.append(mdict)
            # print(f'\nGarment: {garment_name}')
            # for k, v in mdict.items():
            #     if type(v) == dict:
            #         for kk, vv in v.items():
            #             print(k, kk, vv.shape)
            #     else:
            #         print(k, v.shape)


        stacked_material_dict = self._stack_material_dicts(material_dicts)


        example = sample.get_example(0)
        example = self._apply_nodes_multigarment(example, stacked_material_dict)

        for _, edge_label, _ in example.edge_types:
            if edge_label.startswith('coarse') or 'repulsion' in edge_label or 'attraction' in edge_label or 'mesh' in edge_label:
                example = self._apply_edges_multigarment(example, edge_label, stacked_material_dict)


        sample = Batch.from_data_list([example])
        return sample

    def apply_material(self, sample):
        gname = sample['garment_name'][0]
        if ',' not in gname:

            if gname not in self.materials and '::' in gname:
                gname = gname.split('::')[-1]


            if gname in self.materials:
                sample = self.materials[gname].apply_material(sample)
            elif self.mcfg.transfer_missing:
                gname_transfer = '10006_outer'
                print(f'[WARNING] Missing material for {gname}, transferring from {gname_transfer}')
                mdict = self.materials['10006_outer'].transfer_material(sample, 0)

        else:
            sample = self._apply_material_multigarment(sample)

        return sample
    
        
    def _state_dict_to_material_dict(self, state_dict):
        material_dict = defaultdict(dict)
        for key, value in state_dict.items():
            if key.startswith('materials.'):

                key_split = key.split('.')
                garment_name = key_split[1]
                material_key = '.'.join(key_split[2:])
                # _, garment_name, material_key = key.split('.')
                material_dict[garment_name][material_key] = value
        return material_dict
    
    def load_state_dict(self, state_dict):
        material_dict = self._state_dict_to_material_dict(state_dict)

        for garment_name, material_state_dict in material_dict.items():
            material_module = self.material_class(self.material_config, state_dict=material_state_dict)
            self.materials[garment_name] = material_module
        pass
    
    

def create_optimizer(module, config):
    optimizer = torch.optim.Adam(module.parameters(), lr=config.lr)
    return optimizer 