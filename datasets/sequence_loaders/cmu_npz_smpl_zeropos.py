

import os
import pickle
import numpy as np

from utils.common import separate_arms


class SequenceLoader():
    def __init__(self, mcfg, data_root, **kwargs):
        self.mcfg = mcfg

        if mcfg.model_type != 'smpl':
            raise AttributeError("Only SMPL model is supported for "
                                 "datasets.sequence_loaders.cmu_npz_smpl_zeropos,"
                                " would you like to use datasets.sequence_loaders.cmu_npz_smplx_zeropos instead?")
        
        

    def load_sequence(self, **kwargs) -> dict:
        """
        Create a sequence of canonical SMPL bodies parameters 

        :return: dict with SMPL parameters:
            sequence['body_pose'] np.array [Nx63]
            sequence['global_orient'] np.array [Nx3]
            sequence['transl'] np.array [Nx3]
            sequence['betas'] np.array [10]
        """

        N = self.mcfg.n_frames + 2

        sequence = dict()
        sequence['body_pose'] = np.zeros((N, 69))
        sequence['global_orient'] = np.zeros((N, 3))
        sequence['transl'] = np.zeros((N, 3))
        sequence['betas'] = np.zeros(10)
        sequence['subsample'] = 1

        return sequence
    

