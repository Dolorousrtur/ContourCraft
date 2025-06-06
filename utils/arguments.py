from copy import deepcopy
import importlib
import os
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf, DictConfig

from utils.dataloader import DataloaderModule
from utils.defaults import DEFAULTS
import torch


@dataclass
class ExperimentConfig:
    name: Optional[str] = None              # name for the experiment
    save_checkpoint_every: int = 1000     # save checkpoint every n iterations
    save_checkpoint_every_wlong: int = 200     
    n_steps_only_short: int = 50000
    n_epochs: int = 200                     # number of epochs
    max_iter: Optional[int] = None          # max number of iterations
    initial_ts: float = 1/3
    regular_ts: float = 1/30
    
    enable_attractions: bool = False
    

@dataclass
class DataConfig:
    num_workers: int = 0                   # number of workers for dataloader
    batch_size: int = 1                    # batch size (only 1 is supported)
    copy_from: Optional[str] = None        # copy data from another dataloader

@dataclass
class RestartConfig:
    checkpoint_path: Optional[str] = None  
    step_start: Optional[int] = None  
    load_optimizer: bool = True

@dataclass
class MainConfig:
    config: Optional[str] = None           # name of the config file relative to $DEFAULTS.project_dir/configs (without .yaml)

    device: str = 'cuda:0'                 # device to use
    dataloader: DataConfig = DataConfig()
    experiment: ExperimentConfig = ExperimentConfig()
    restart: RestartConfig = RestartConfig()
    detect_anomaly: bool = False           # torch.autograd.detect_anomaly
    step_start: int = 0                    


def struct_fix(config):
    OmegaConf.set_struct(config, False)
    for k, v in config.items():
        if type(v) == DictConfig:
            struct_fix(v)



def load_module(module_type: str, module_config: DictConfig, module_name: str = None):
    """
    This function loads a singular module from OmegaConf config.
    It also merges the default config of the module with the config from the config file.

    :param module_type: type of module to load (e.g. models, runners, etc.)
    :param module_config: OmegaConf config
    :param module_name: module name
    :return: loaded python module
    """
    # if module_name is not specified, take the first OmegaConf key
    if module_name is None:
        module_name = list(module_config.keys())[0]

    # load python module
    module = importlib.import_module(f'{module_type}.{module_name}')

    # load default config from module.Config and merge with config from config file
    default_module_config = OmegaConf.create(module.Config)
    OmegaConf.set_struct(default_module_config, False)

    if module_config[module_name] is None:
        module_config[module_name] = default_module_config
    else:
        module_config[module_name] = OmegaConf.merge(default_module_config, module_config[module_name])
    return module

def load_dataset_params(conf):
    dataset_modules = {}

    for dataloader_name, dataloader_conf in conf.dataloaders.items():

        if 'copy_from' in dataloader_conf and dataloader_conf.copy_from is not None:
            new_cfg = deepcopy(conf.dataloaders[dataloader_conf.copy_from])
            new_cfg = OmegaConf.merge(new_cfg, dataloader_conf)
            conf.dataloaders[dataloader_name] = new_cfg

        dataset_module = load_module('datasets', dataloader_conf.dataset)
        dataset_modules[dataloader_name] = dataset_module

    return dataset_modules, conf

def load_criterion_dicts(conf):
    criterion_dicts = {}

    for criterion_dict_name, criterion_conf in conf.criterions.items():
        criterion_dict = {}
        criterion_names = criterion_conf.keys()
        # Can have arbitrary number of criterions
        for criterion_name in criterion_names:
            criterion_module = load_module('criterions', criterion_conf, criterion_name)
            criterion_dict[criterion_name] = criterion_module
        criterion_dicts[criterion_dict_name] = criterion_dict
    return criterion_dicts

def load_params(config_name: str=None, config_dir: str=None):
    """
    Build OmegaConf config and the modules from the config file.
    :param config_name: name of the config file (without .yaml)
    :param config_dir: root directory of the config files
    :return:
        modules: dict of loaded modules
        conf: OmegaConf config
    """

    # Set default config directory
    if config_dir is None:
        config_dir = Path(DEFAULTS.project_dir) / 'configs'

    # Load default config from MainConfig and merge in cli parameters
    conf = OmegaConf.structured(MainConfig)
    struct_fix(conf)

    conf_cli = OmegaConf.from_cli()
    if config_name is None:
        config_name = conf_cli.config
        conf = OmegaConf.merge(conf, conf_cli)

    # Load config file and merge it in
    conf['config'] = config_name
    config_path = os.path.join(config_dir, config_name + '.yaml')
    conf_file = OmegaConf.load(config_path)
    OmegaConf.set_struct(conf, False)
    OmegaConf.set_struct(conf_file, False)
    conf = OmegaConf.merge(conf, conf_file)
    conf = OmegaConf.merge(conf, conf_cli)

    # Load modules from config
    modules = {}

    modules['model'] = load_module('models', conf.model)
    modules['runner'] = load_module('runners', conf.runner)

    modules['criterion_dicts'] = load_criterion_dicts(conf)



    if 'material_stack' in conf:
        modules['material_stack'] = load_module('material', conf.material_stack)

    modules['datasets'], conf, = load_dataset_params(conf)

    return modules, conf


def create_module(module, module_config: DictConfig, module_name: str=None):
    """
    Create a module object from the python module and the config file.
    :param module: python module (should have `create` method)
    :param module_config: OmegaConf config for the module
    :param module_name: name of the module
    :return: module object: loaded module object
    """
    if module_name is None:
        module_name = list(module_config.keys())[0]
    module_config = module_config[module_name]
    module_object = module.create(module_config)
    return module_object


def create_criterion_dicts(modules: dict, config: DictConfig):
    criterion_dicts_dict = {}

    for criterion_dict_name, criterion_dict in modules['criterion_dicts'].items():
        criterion_dict_object = {}
        for criterion_name, criterion_module in criterion_dict.items():
            criterion = create_module(criterion_module, config['criterions'][criterion_dict_name],
                                      module_name=criterion_name)
            if hasattr(criterion, 'name'):
                criterion_name = criterion.name
            criterion_dict_object[criterion_name] = criterion
        criterion_dicts_dict[criterion_dict_name] = criterion_dict_object

    return criterion_dicts_dict


def create_runner(modules: dict, config: DictConfig, create_aux_modules=True):
    """
    Create a runner object from the specified runner module.
    :param modules: dict of loaded .py modules
    :param config: OmegaConf config
    :param create_aux_modules: whether to create optimizer and scheduler
    :return: runner_module: .py runner module
    :return: runner: Runner object
    :return aux_modules: dict of auxiliary modules (optimizer, scheduler, etc.)
    """
    runner_module = modules['runner']
    runner_name = list(config['runner'].keys())[0]
    runner_config = config['runner'][runner_name]

    # create model object
    model = create_module(modules['model'], config['model'])

    criterion_dicts = create_criterion_dicts(modules, config)

    # create Runner object from the specified runner module
    runner = runner_module.Runner(model, criterion_dicts, runner_config)

    # create optimizer and scheduler from the specified runner module
    aux_modules = dict()

    if create_aux_modules:
        optimizer, scheduler = runner_module.create_optimizer(runner, runner_config.optimizer)
        aux_modules['optimizer'] = optimizer
        aux_modules['scheduler'] = scheduler

    
    if 'material_stack' in modules:
        material_stack = create_module(modules['material_stack'], config.material_stack)
        aux_modules['material_stack'] = material_stack

    return runner_module, runner, aux_modules


def create_dataloader_modules(modules: dict, config: DictConfig):
    """
    Create a dataloader module.
    :param modules:
    :param config: OmegaConf config
    :return: DataloaderModule
    """

    dataloader_modules = {}

    for dataset_name, dataset_module in modules['datasets'].items():
        dataset_cfg = config.dataloaders[dataset_name].dataset
        dataset = create_module(dataset_module, dataset_cfg)
        dataloader_m = DataloaderModule(dataset, config['dataloader'])
        dataloader_modules[dataset_name] = dataloader_m

    return dataloader_modules


def create_modules(modules: dict, config: DictConfig, create_aux_modules: bool=True):
    """
    Create all the modules from the config file.
    :param modules: dict of loaded python modules
    :param config: full OmegaConf config
    :param create_aux_modules: whether to create optimizer and scheduler
    :return:
        dataloader_m: dataloader module (should have `create_dataloader` method)
        runner: runner module
        training_module: Runner object from the selected runner
        aux_modules: dict of auxiliary modules (e.g. optimizer, scheduler)
    """

    runner_module, runner, aux_modules = create_runner(modules, config, create_aux_modules=create_aux_modules)
    dataloader_ms = create_dataloader_modules(modules, config)
    return dataloader_ms, runner_module, runner, aux_modules

def load_from_checkpoint(cfg, runner, aux_modules):

    if cfg.restart.checkpoint_path is None:
        return runner, aux_modules
    
    checkpoint_path = Path(DEFAULTS.data_root) / cfg.restart.checkpoint_path

    if not os.path.exists(checkpoint_path):
        return runner, aux_modules

    sd = torch.load(checkpoint_path)
    runner.load_state_dict(sd['training_module'])

    print(f'LOADED CHECKPOINT FROM {checkpoint_path}')

    if cfg.restart.step_start is not None:
        cfg.step_start = cfg.restart.step_start

    if cfg.restart.load_optimizer:

        base_lrs = [group['initial_lr'] for group in aux_modules['optimizer'].param_groups]
        for k, v in aux_modules.items():
            if k in sd:
                print(f'{k} LOADED!')
                v.load_state_dict(sd[k])

        if 'scheduler' in aux_modules:
            aux_modules['scheduler'].base_lrs = base_lrs

    return runner, aux_modules
