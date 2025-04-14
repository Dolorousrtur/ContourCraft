

from utils.arguments import load_params
from utils.common import move2device
from utils.dataloader import DataloaderModule
from utils.validation import apply_material_params, create_one_sequence_dataloader, create_postcvpr_one_sequence_dataloader, load_runner_from_checkpoint

from datasets.zeropos_init import Config as ZPDatasetConfig
from datasets.zeropos_init import create as zp_create_dataset

class Simulator:
    def __init__(self, checkpoint_path, sequence_loader='cmu_npz_smplx',
                  config_name='contourcraft', material_dict=None):
        
        
        # load the config from .yaml file and load .py modules specified there
        modules, experiment_config = load_params(config_name)

        # modify the config to use it in validation 
        material_dict = material_dict or self._default_material_dict()
        experiment_config = apply_material_params(experiment_config, material_dict)

        # load Runner object and the .py module it is declared in
        runner_module, runner = load_runner_from_checkpoint(checkpoint_path, modules, experiment_config)
        self.runner = runner

        self.sequence_loader = sequence_loader

    def _default_material_dict(self):
        material_dict = dict()
        material_dict['density'] = 0.20022
        material_dict['lame_mu'] = 23600.0
        material_dict['lame_lambda'] = 44400
        material_dict['bending_coeff'] = 3.962e-05
        return material_dict


    def _run_sequence(self, sample, n_steps=-1):
        # sample = move2device(sample, 'cuda:0')
        trajectories_dict = self.runner.valid_rollout(sample,  bare=True, n_steps=n_steps)
        return trajectories_dict


    # def run(self, sequence_path, garment_dict_file, garment_name, n_steps=-1, **kwargs):
    #     dataloader = create_postcvpr_one_sequence_dataloader(sequence_path, garment_name, 
    #                                                 garment_dict_file, sequence_loader=self.sequence_loader, 
    #                                                 obstacle_dict_file=None, **kwargs)
        
    #     sample = next(iter(dataloader))


    #     trajectories_dict = self._run_sequence(sample, n_steps)

    #     return trajectories_dict
    
    def _create_zeropos_sample(self, garment_name, n_frames=200, **kwargs):
        
        config = 'aux/untangle'
        dataloader = create_one_sequence_dataloader(use_config=config, data_root='', single_sequence_file=None,
                                                    single_sequence_garment=garment_name,
                                                    sequence_loader='cmu_npz_smplx_zeropos', n_frames=n_frames, **kwargs)

        sample = next(iter(dataloader))
        return sample
    
    def run_zeropos(self, garment_name, n_steps=200, gender='female'):
        sample = self._create_zeropos_sample(garment_name, n_steps, gender=gender)


        trajectories_dict = self._run_sequence(sample)

        return trajectories_dict