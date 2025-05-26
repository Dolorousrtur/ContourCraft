import os
import socket

from munch import munchify

hostname = socket.gethostname()

DEFAULTS = dict()

DEFAULTS['project_name'] = 'ccraft'


DEFAULTS['data_root'] = '/path/to/ccraft_data'
DEFAULTS['aux_data'] = os.path.join(DEFAULTS['data_root'], 'aux_data')
DEFAULTS['project_dir'] = 'path/to/this/repo'
DEFAULTS['experiment_root'] = os.path.join(DEFAULTS['data_root'], 'experiments')

DEFAULTS['CMU_root'] = '/path/to/AMASS/smpl/CMU'

DEFAULTS = munchify(DEFAULTS)
