import os

import numpy as np
import torch

from utils.arguments import load_from_checkpoint, load_params, create_modules
from utils.writer import WandbWriter


def main():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    modules, config = load_params()
    dataloader_ms, runner_module, runner, aux_modules = create_modules(modules, config)

    runner, aux_modules = load_from_checkpoint(config, runner, aux_modules)

    if config.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        
    writer = WandbWriter(config)
    if config.experiment.use_writer:
        writer = WandbWriter(config)
    else:
        writer = None

    global_step = config.step_start

    print('config.restart.step_start', config.restart.step_start)
    print('config.step_start', config.step_start)
    print('global_step', global_step)
    assert False

    torch.manual_seed(57)
    np.random.seed(57)
    for i in range(config.experiment.n_epochs):
        dataloaders_dict = dict()

        for dataloader_name, dataloader in dataloader_ms.items():
            dataloaders_dict[dataloader_name] = dataloader.create_dataloader()

        global_step = runner_module.run_epoch(runner, aux_modules, dataloaders_dict, config, writer,
                                       global_step=global_step)

        if config.experiment.max_iter is not None and global_step > config.experiment.max_iter:
            break


if __name__ == '__main__':
    main()
    
