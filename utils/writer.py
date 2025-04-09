import os
from typing import Dict

import numpy as np
import wandb
from omegaconf import OmegaConf

from utils.defaults import DEFAULTS


class Writer:
    def __init__(self, mcfg):
        pass

    def write_dict(self, dict_to_write: Dict[str, float]):
        pass


def collect_config(config_dict, prefix=''):
    flat_dict = dict()

    for k, v in config_dict.items():
        full_key = f"{prefix}{k}"
        if type(v) == dict:
            flat_dict[f'{full_key}.name(s)'] = ','.join(v.keys())
            flat_dict.update(collect_config(v, prefix=f'{full_key}.'))
        else:
            flat_dict[full_key] = v

    return flat_dict


class WandbWriter(Writer):
    def __init__(self, mcfg):
        super().__init__(mcfg)
        os.makedirs(DEFAULTS.experiment_root, exist_ok=True)
        flat_config = collect_config(OmegaConf.to_container(mcfg))
        project = DEFAULTS.project_name if mcfg.experiment.use_writer else "dummy"

        try:
            api = wandb.Api()
            runs = api.runs(f'agrigorev/{project}')
            numbers = [int(x.name.split('-')[-1]) for x in runs if '-' in x.name]
            if len(numbers) == 0:
                new_number = 0
            else:
                new_number = np.max(numbers) + 1

            name = mcfg.config.split('/')[-1]
            name += f"-{new_number}"
        except:

            name = mcfg.config
            name += f"-{0}"

        wandb.init(name=name, project=project, entity="agrigorev", dir=DEFAULTS.experiment_root, config=flat_config)

        mcfg.run_dir = os.path.dirname(wandb.run.dir)

        if mcfg.step_start is not None:
            wandb.log({}, step=mcfg.step_start)

    def write_dict(self, dict_to_write: Dict[str, float], step=None):

        # for k, v in dict_to_write.items():
        #     dict_to_write[k] = 0.
        wandb.log(dict_to_write, step=step)
