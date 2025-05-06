# wbody2 == mimp_cut_wbody_omw_ombp
import os
from pathlib import Path
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import torch
from omegaconf import II
from omegaconf.dictconfig import DictConfig
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData, Batch
from tqdm import tqdm

from runners.utils.collector import SampleCollector
from runners.utils.collision import CollisionPreprocessor
from runners.utils.impulses import CollisionSolver
from runners.utils.material import RandomMaterial
from utils.cloth_and_material import ClothNoMat, FaceNormals, ClothMatAug
from utils.common import move2device, save_checkpoint, add_field_to_pyg_batch, copy_pyg_batch, TorchTimer, NodeType
from utils.defaults import DEFAULTS


@dataclass
class MaterialConfig:
    density_min: float = 0.20022
    density_max: float = 0.20022
    lame_mu_min: float = 23600.0
    lame_mu_max: float = 23600.0
    lame_lambda_min: float = 44400
    lame_lambda_max: float = 44400
    bending_coeff_min: float = 3.9625778333333325e-05
    bending_coeff_max: float = 3.9625778333333325e-05
    bending_multiplier: float = 1.

    density_override: Optional[float] = None
    lame_mu_override: Optional[float] = None
    lame_lambda_override: Optional[float] = None
    bending_coeff_override: Optional[float] = None


@dataclass
class OptimConfig:
    lr: float = 1e-4
    decay_rate: float = 1e-1
    decay_min: float = 0
    decay_steps: int = 5_000_000
    step_start: int = 0


@dataclass
class SafecheckConfig:
    n_impulse_iters: int = 10
    max_riz_size: int = 30
    n_rest2pos_steps: int = 10
    riz_max_mp_steps: int = 10
    riz_max_steps_total: int = 100
    pinned_mass: float = 1e9
    riz_epsilon: float = 1e-10
    max_ncoll: int = -1
    max_impulse_norm: Optional[float] = None
    double_precision_impulse: bool = True
    double_precision_riz: bool = True
    device: str = II('device')


@dataclass
class Config:
    optimizer: OptimConfig = OptimConfig()
    material: MaterialConfig = MaterialConfig()
    safecheck: SafecheckConfig = SafecheckConfig()
    warmup_steps: int = 100
    increase_roll_every: int = 5000
    roll_max: int = 5
    push_eps: float = 2e-3
    grad_clip: Optional[float] = 1.
    overwrite_pos_every_step: bool = False
    cutout_with_attractions: bool = False
    long_rollout_steps: int = -1

    ft_resolve_target: bool = True
    reset_every: int = 10

    enable_attractions: bool = True


    initial_ts: float = II("experiment.initial_ts")
    regular_ts: float = II("experiment.regular_ts")


class Runner(nn.Module):
    def __init__(self, model: nn.Module, criterion_dicts: dict,
                 mcfg: DictConfig):
        super().__init__()

        self.model = model
        self.criterion_dict_hood = criterion_dicts['ccraft']
        self.criterion_dict_ft = criterion_dicts['finetune']
        self.mcfg = mcfg

        self.cloth_obj = ClothMatAug(None, always_overwrite_mass=True)
        self.cloth_obj_nomat = ClothNoMat()
        self.normals_f = FaceNormals()

        self.sample_collector = SampleCollector(mcfg, obstacle=True)
        self.random_material = RandomMaterial(mcfg.material)
        self.safecheck_solver = CollisionSolver(mcfg.safecheck)
        self.collision_solver = CollisionPreprocessor(mcfg)


    def valid_rollout(self, sequence, material_stack, n_steps=-1, bare=False, record_time=False, progressbar=True):

        record_time = True

        ft = material_stack is not None


        if 'iter' not in sequence['cloth']:
            sequence = add_field_to_pyg_batch(sequence, 'iter', [0], 'cloth', reference_key=None)
        sequence = self._add_cloth_obj(sequence, ft=ft)

        is_obstacle = 'obstacle' in sequence.node_types
        if is_obstacle:
            n_samples = sequence['obstacle'].lookup.shape[1]
        else:
            n_samples = sequence['cloth'].lookup.shape[1]

        if n_steps >= 0:
            n_samples = min(n_samples, n_steps)

        trajectories_dicts = defaultdict(list)

        if record_time:
            st_time = time.time()

        st = 0
        self.model.eval()
        trajectories, metrics_dict = self._rollout(sequence, material_stack, st, n_samples - st,
                                                    progressbar=progressbar, bare=bare)

        if record_time:
            total_time = time.time() - st_time
            metrics_dict['time'] = total_time

        # trajectories_dicts['pred'] = trajectory
            
        cloth_faces = sequence['cloth'].faces_batch.T.cpu().numpy()
        

        trajectories_dicts.update(trajectories)
        trajectories_dicts['metrics'] = dict(metrics_dict)
        trajectories_dicts['cloth_faces'] = cloth_faces

        if 'garment_id' in sequence['cloth']:
            trajectories_dicts['garment_id'] = sequence['cloth'].garment_id.cpu().numpy()
            
        trajectories_dicts['vertex_type'] = sequence['cloth'].vertex_type.cpu().numpy()

        if 'uv_coords' in sequence['cloth']:
            uv_faces = sequence['cloth'].uv_faces_batch.T.cpu().numpy()
            trajectories_dicts['uv_coords'] = sequence['cloth'].uv_coords.cpu().numpy()
            trajectories_dicts['uv_faces'] = uv_faces

        if is_obstacle:
            trajectories_dicts['obstacle_faces'] = sequence['obstacle'].faces_batch.T.cpu().numpy()

        for s in ['pred', 'obstacle']:
            if s in trajectories_dicts:
                trajectories_dicts[s] = torch.stack(trajectories_dicts[s], dim=0).cpu().numpy()
        return trajectories_dicts

    def _rollout(self, sample, material_stack, start_idx, n_steps, progressbar=True, bare=False):
        ft = material_stack is not None
        trajectories = defaultdict(list)
        is_obstacle = 'obstacle' in sample.node_types

        metrics_dict = defaultdict(list)

        if n_steps == 0:
            pbar = range(start_idx, start_idx + 1)
        else:
            pbar = range(start_idx, start_idx + n_steps)
        if progressbar:
            pbar = tqdm(pbar)


        prev_out_sample = None

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        sample = self.prepare_sample(sample, ft=ft)

        for i in pbar:
            sample_step = self.collect_sample(sample, i, prev_out_sample)

            if i == 0:
                sample_step, sample = self.update_sample_1st_step(sample_step, sample)

            if i == 0:
                trajectories['pred'].append(sample_step['cloth'].prev_pos)
                trajectories['pred'].append(sample_step['cloth'].pos)

                if is_obstacle:
                    trajectories['obstacle'].append(sample_step['obstacle'].prev_pos)
                    trajectories['obstacle'].append(sample_step['obstacle'].pos)


            if n_steps == 0:
                break


            with TorchTimer(metrics_dict, 'hood_time', start=start, end=end):
                sample_step = self.model(sample_step, material=material_stack)



            prev_out_sample = sample_step.detach()

            trajectories['pred'].append(prev_out_sample['cloth'].pred_pos)
            trajectories['gt'].append(prev_out_sample['cloth'].target_pos)
            if is_obstacle:
                trajectories['obstacle'].append(prev_out_sample['obstacle'].target_pos)

            if not bare:
                loss_dict_hood, loss_weight_dict_hood, _, _ = self.criterion_pass(sample_step, self.criterion_dict_ft)
                metrics_dict = self.add_metrics(loss_dict_hood, loss_weight_dict_hood,
                                                metrics_dict, 'valid/')
                
                metrics_dict = self.add_edge_scale_to_metrics(metrics_dict, sample_step, prefix='valid/')
                

        metrics_dict = self.aggregate_metrics_dict(metrics_dict)

        return trajectories, metrics_dict



    def set_random_material(self, sample):
        sample, self.cloth_obj = self.random_material.add_material(sample, self.cloth_obj)
        return sample

    def _add_cloth_obj(self, sample, ft=False):
        if  ft:
            sample = self.cloth_obj_nomat.set_batch(sample, overwrite_pos=self.mcfg.overwrite_pos_every_step)
            sample['cloth'].cloth_obj = self.cloth_obj_nomat
        else:
            sample = self.set_random_material(sample)
            sample = self.cloth_obj.set_batch(sample, overwrite_pos=self.mcfg.overwrite_pos_every_step)
            sample['cloth'].cloth_obj = self.cloth_obj
        return sample

    def criterion_pass(self, sample_step, criterion_dict):
        sample_step.cloth_obj = self.cloth_obj
        loss_dict = dict()
        gradient_dict = dict()
        loss_weight_dict = dict()
        metrics_dict = dict()
        for criterion_name, criterion in criterion_dict.items():
            ld = criterion(sample_step)
            for k, v in ld.items():
                if 'loss' in k:
                    loss_dict[f"{criterion_name}_{k}"] = v
                elif 'gradient' in k:
                    gradient_dict[f"{criterion_name}_{k}"] = v
                elif 'weight' in k:
                    loss_weight_dict[f"{criterion_name}_{k}"] = v
                else:
                    metrics_dict[f"{criterion_name}_{k}"] = v

        return loss_dict, loss_weight_dict, gradient_dict, metrics_dict


    def collect_sample(self, sample, idx, prev_out_dict=None, random_ts=False, is_short=False):
        sample_step = copy_pyg_batch(sample)

        # coly fields from the previous step (pred_pos -> pos, pos->prev_pos)
        sample_step = self.sample_collector.copy_from_prev(sample_step, prev_out_dict)
        ts = self.mcfg.regular_ts

        if idx > 0:
            sample_step = self.sample_collector.lookup2target(sample_step, idx)

        is_init = False
        if is_short and idx == 0:
            if random_ts:
                is_init = np.random.rand() > 0.5
                if is_init:
                    ts = self.mcfg.initial_ts
            else:
                is_init = True
                ts = self.mcfg.initial_ts

        sample_step = self.sample_collector.add_is_init(sample_step, is_init)
        sample_step = self.sample_collector.add_timestep(sample_step, ts)
        sample_step = self.sample_collector.add_velocity(sample_step, prev_out_dict)
        return sample_step

    def add_metrics_from_dict(self, loss_dict, loss_weight_dict, metrics_dict_step, prefix):
        for k, v in loss_dict.items():
            k_weight = k.replace('loss', 'weight')
            v = v.item()


            weight = loss_weight_dict[k_weight] if k_weight in loss_weight_dict else 1
            v = v / weight if weight != 0 else 0

            k_pref = prefix + k
            metrics_dict_step[k_pref] = v

        return metrics_dict_step

    def add_metrics_ratios(self, metrics_dict_step, prefix_from, prefix_to, remove_to=False):
        keys_to = [k for k in metrics_dict_step if k.startswith(prefix_to)]

        for key_to in keys_to:
            v_to = metrics_dict_step[key_to]
            key_from = key_to.replace(prefix_to, prefix_from)

            if key_from not in metrics_dict_step:
                continue

            v_from = metrics_dict_step[key_from]
            ratio = v_to / v_from

            key_ratio = key_to.replace(prefix_to, 'ratio/')
            metrics_dict_step[key_ratio] = ratio

            if remove_to:
                del metrics_dict_step[key_to]

        return metrics_dict_step

    def add_metrics(self, loss_dict_hood, loss_weight_dict_hood,
                    metrics_dict, prefix=None):
        
        if prefix is None:
            prefix = ''

        metrics_dict_step = {}
        metrics_dict_step = self.add_metrics_from_dict(loss_dict_hood, loss_weight_dict_hood, metrics_dict_step,
                                                       prefix=prefix)


        for k, v in metrics_dict_step.items():
            metrics_dict[k].append(v)

        return metrics_dict

    def collect_penetrating_mask(self, sample_step, sample):
        penetrating_mask = torch.zeros_like(sample_step['cloth'].pos[:, :1]).bool()
        sample_step = add_field_to_pyg_batch(sample_step, 'penetrating_mask', penetrating_mask,
                                             'cloth',
                                             reference_key='pos')
        sample = add_field_to_pyg_batch(sample, 'penetrating_mask', penetrating_mask,
                                        'cloth',
                                        reference_key='pos')
        return sample_step, sample


    
    def add_gradient(self, sample_step, optimizer, gradient, weight=1.):
        params = optimizer.param_groups[0]['params']
        params = [param for param in params if param.requires_grad]

        model_icontour_grad = torch.autograd.grad(sample_step['cloth'].pred_pos, params, grad_outputs=gradient, allow_unused=True, retain_graph=True)

        i = 0
        for param in optimizer.param_groups[0]['params']:
            if param.requires_grad:
                if model_icontour_grad[i] is not None:
                    if param.grad is None:
                        param.grad = torch.zeros_like(param)
                    param.grad += model_icontour_grad[i]*weight
                i += 1

    def zerograd2none(self, material_stack):
        if material_stack is None:
            return

        for parameter in material_stack.parameters():
            if parameter.grad is not None and parameter.grad.pow(2).sum() == 0:
                parameter.grad = None

    def optimizer_step(self, optimizer_list, scheduler_list, loss_dict, gradient_dict, sample_step, material_stack=None):

        if optimizer_list is not None:
            # for group in optimizer.param_groups:
            #     print(group['lr'])
            loss = sum(loss_dict.values())

            for optimizer in optimizer_list:
                optimizer.zero_grad()


                for gradient_name, gradient in gradient_dict.items():
                    # print(gradient_name, gradient.abs().sum())
                    self.add_gradient(sample_step, optimizer, gradient, weight=1.)

            loss.backward()

                

            if self.mcfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.mcfg.grad_clip)

            self.zerograd2none(material_stack)

            for optimizer in optimizer_list:
                optimizer.step()

            if scheduler_list is not None:
                for scheduler in scheduler_list:
                    if scheduler is not None:
                        scheduler.step()

    def prepare_sample(self, sample, ft=False):
        sample = self._add_cloth_obj(sample, ft=ft)

        if not ft:
            sample = self.safecheck_solver.mark_penetrating_faces(sample, dummy=True)
        return sample

    def update_sample_1st_step(self, sample_step, sample):
        sample_step, sample = self.collect_penetrating_mask(sample_step, sample)
        return sample_step, sample

    def aggregate_metrics_dict(self, metrics_dict):
        metrics_dict_agg = {}
        for k, v in metrics_dict.items():
            if k == 'riz_itersmax_riz_size':
                v = np.max(v)
                v = min(v, self.mcfg.safecheck.max_riz_size)
            else:
                v = np.mean(v)

            metrics_dict_agg[k] = v
        return metrics_dict_agg


    def forward_long(self, sample, optimizer_list=None, scheduler_list=None) -> dict:
        sample = self.prepare_sample(sample)
        roll_steps = sample['cloth'].lookup.shape[1]

        metrics_dict = defaultdict(list)
        prev_out_sample = None
        _i = 0

        if self.mcfg.long_rollout_steps >= 0:
            roll_steps = min(self.mcfg.long_rollout_steps, roll_steps)

        for i in range(roll_steps):
            sample = add_field_to_pyg_batch(sample, 'step', [i], 'cloth', reference_key=None)

            sample = self._add_cloth_obj(sample)
            sample_step = self.collect_sample(sample, i, prev_out_sample)
            sample_step = self.safecheck_solver.mark_penetrating_faces(sample_step, object='obstacle', use_target=True) 

            if i == 0:
                sample_step = self.collision_solver.solve(sample_step)
                sample_step, sample = self.update_sample_1st_step(sample_step, sample)


            self.model.train(False)
            sample_step = self.model(sample_step, world_edges=True, fake_icontour=False)
            loss_dict_hood, loss_weight_dict_hood, gradient_dict, _ = self.criterion_pass(sample_step, self.criterion_dict_hood)

            self.optimizer_step(optimizer_list, scheduler_list, loss_dict_hood, gradient_dict, sample_step)

            prev_out_sample = sample_step.detach()
            _i = i + 1

            metrics_dict = self.add_metrics(loss_dict_hood, loss_weight_dict_hood,
                                            metrics_dict, prefix='long/')

        metrics_dict = self.aggregate_metrics_dict(metrics_dict)
        return metrics_dict

    def add_edge_scale_to_metrics(self, metrics_dict, sample_step, prefix=''):
        pred_pos = sample_step['cloth'].pred_pos
        target_pos = sample_step['cloth'].target_pos

        edge_index = sample_step['cloth', 'mesh_edge', 'cloth'].edge_index.T

        edges_pred = pred_pos[edge_index[0]] - pred_pos[edge_index[1]]
        edges_target = target_pos[edge_index[0]] - target_pos[edge_index[1]]

        edges_target_norm = edges_target.norm(dim=-1)
        edges_pred_norm = edges_pred.norm(dim=-1)
        edge_scale = edges_pred_norm / edges_target_norm
        edge_scale = edge_scale[edge_scale == edge_scale].mean().item()

        metrics_dict[prefix+'edge_scale'].append(edge_scale)
        return metrics_dict


    def forward_ft(self, sample, material_stack, optimizer_list=None, scheduler_list=None) -> dict:

        sample = self.prepare_sample(sample, ft=True)
        roll_steps = sample['cloth'].lookup.shape[1]

        metrics_dict = defaultdict(list)
        prev_out_sample = None

        trajectories = defaultdict(list)
        trajectories['cloth_faces'] = sample['cloth'].faces_batch.T.cpu().numpy()
        trajectories['obstacle_faces'] = sample['obstacle'].faces_batch.T.cpu().numpy()

        reset_every = self.mcfg.reset_every
        if reset_every > 0:
            random_shift = random.randint(0, reset_every)
        else:
            random_shift = 0

        for i in range(roll_steps):
            sample = add_field_to_pyg_batch(sample, 'step', [i], 'cloth', reference_key=None)
            sample = self._add_cloth_obj(sample, ft=True)



            if reset_every > 0 and (i+random_shift) % reset_every == 0:
                prev = None
            else:
                prev = prev_out_sample            
             
            sample_step = self.collect_sample(sample, i, prev)


            sample_step = self.safecheck_solver.mark_penetrating_faces(sample_step, object='obstacle', use_target=True) 


            sample_step = self.collision_solver.solve(sample_step, target=self.mcfg.ft_resolve_target) 


            self.model.train(False)
            sample_step = self.model(sample_step, world_edges=True, fake_icontour=False, material=material_stack)
            loss_dict_hood, loss_weight_dict_hood, gradient_dict, _ = self.criterion_pass(sample_step, self.criterion_dict_ft)


            # loss_dict_impulse, loss_weight_dict_impulse = None, None
            self.optimizer_step(optimizer_list, scheduler_list, loss_dict_hood, gradient_dict, sample_step, material_stack=material_stack)


            trajectories['pred'].append(sample_step['cloth'].pred_pos.detach().cpu().numpy())
            trajectories['obstacle'].append(sample_step['obstacle'].target_pos.detach().cpu().numpy())

            prev_out_sample = sample_step.detach()

            metrics_dict = self.add_metrics(loss_dict_hood, loss_weight_dict_hood,
                                            metrics_dict, prefix='ft/')
            metrics_dict = self.add_edge_scale_to_metrics(metrics_dict, sample_step, prefix='ft/')

        metrics_dict = self.aggregate_metrics_dict(metrics_dict)

        trajectories['pred'] = np.stack(trajectories['pred'], axis=0)
        trajectories['obstacle'] = np.stack(trajectories['obstacle'], axis=0)

        return metrics_dict
    


def create_optimizer(training_module: Runner, mcfg: DictConfig):
    optimizer = Adam(training_module.model.parameters(), lr=mcfg.lr)

    def sched_fun(step):
        decay = mcfg.decay_rate ** (step // mcfg.decay_steps)
        decay = max(decay, mcfg.decay_min)

        # print(step, decay)
        return decay

    scheduler = LambdaLR(optimizer, sched_fun)
    scheduler.last_epoch = mcfg.step_start

    return optimizer, scheduler


def compute_epoch_size(dataloader_long, dataloader_ft, cfg, global_step):
    len_ft = len(dataloader_ft)
    len_long = len(dataloader_long)
       
    total_steps = min(len_long, len_ft)
    total_steps  *= 2

    return total_steps


def step_long(training_module, sample, optimizer_list, scheduler_list):
    ld_to_write = training_module.forward_long(sample, optimizer_list=optimizer_list,
                                               scheduler_list=scheduler_list)
    return ld_to_write

def step_ft(training_module, material_stack, sample, optimizer_list, scheduler_list):
    ld_to_write = training_module.forward_ft(sample, material_stack, optimizer_list=optimizer_list,
                                              scheduler_list=scheduler_list)
    return ld_to_write


def make_checkpoint(runner, aux_modules, cfg, global_step):
    if global_step < cfg.experiment.n_steps_only_short:
        to_save = global_step % cfg.experiment.save_checkpoint_every == 0
    else:
        to_save = global_step % cfg.experiment.save_checkpoint_every_wlong == 0

    if to_save:


        if hasattr(cfg, 'checkpoints_dir') and cfg.checkpoints_dir is not None:
            checkpoints_dir = Path(DEFAULTS.data_root) / cfg.checkpoints_dir
        else:
            now = datetime.now()
            dt_string = now.strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join(DEFAULTS.experiment_root, dt_string)
            checkpoints_dir = os.path.join(run_dir, 'checkpoints')
            cfg.checkpoints_dir = checkpoints_dir

        os.makedirs(checkpoints_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoints_dir, f"step_{global_step:010d}.pth")
        print('Saving checkpoint to', checkpoint_path)

        save_checkpoint(runner, aux_modules, cfg, checkpoint_path)





def run_epoch(runner: Runner, aux_modules: dict, dataloaders_dict: DataLoader,
               cfg: DictConfig, writer=None, global_step=None):
    global_step = global_step or 0
    runner.model.eval() # Don't update collected statistics

    material_stack = aux_modules['material_stack']


    optimizer_model = aux_modules['optimizer']
    scheduler_model = aux_modules['scheduler']

    optimizer_material = aux_modules['optimizer_material']
    scheduler_material = aux_modules['scheduler_material'] if 'scheduler_material' in aux_modules else None

    dataloader_long = dataloaders_dict['long']
    dataloader_ft = dataloaders_dict['finetune']

    n_steps = compute_epoch_size(dataloader_long, dataloader_ft, cfg, global_step)

    if cfg.experiment.max_iter is not None:
        n_steps = min(n_steps, cfg.experiment.max_iter - global_step)

    prbar = tqdm(range(n_steps), desc=cfg.config)

    ft_iter = dataloader_ft.__iter__()
    long_iter = dataloader_long.__iter__()

    last_step = 'long'
    for i in prbar:
        global_step += 1

        if cfg.experiment.max_iter is not None and global_step > cfg.experiment.max_iter:
            break

        if last_step == 'long':
            curr_step = 'ft'
            sample = next(ft_iter)
        else:
            curr_step = 'long'
            sample = next(long_iter)

        

        last_step = curr_step
        sample = move2device(sample, cfg.device)

        B = sample.num_graphs
        sample = add_field_to_pyg_batch(sample, 'iter', [global_step] * B, 'cloth', reference_key=None)

        if curr_step == 'ft':
            optimizer_list = [optimizer_model, optimizer_material]
            scheduler_list = [scheduler_model, scheduler_material]

            ld_to_write = step_ft(runner, material_stack, sample, optimizer_list, scheduler_list)
        elif curr_step == 'long':
            optimizer_list = [optimizer_model]
            scheduler_list = [scheduler_model]
            ld_to_write = step_long(runner, sample, optimizer_list, scheduler_list)
        else:
            raise Exception(f'Wrong step label {curr_step}')

        if writer is not None:
            writer.write_dict(ld_to_write)

        make_checkpoint(runner, aux_modules, cfg, global_step)


    return global_step

