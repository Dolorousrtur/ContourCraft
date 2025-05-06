import os
import socket

from munch import munchify

hostname = socket.gethostname()

DEFAULTS = dict()

DEFAULTS['project_name'] = 'ccraft'


# DEFAULTS['CMU_root'] = '/path/to/AMASS/smpl/CMU'
# DEFAULTS['data_root'] = '/path/to/ccraft_data'
# DEFAULTS['aux_data'] = os.path.join(DEFAULTS['data_root'], 'aux_data')
# DEFAULTS['project_dir'] = 'path/to/this/repo'
# DEFAULTS['experiment_root'] = os.path.join(DEFAULTS['data_root'], 'experiments')

DEFAULTS['CMU_root'] = ''
DEFAULTS['data_root'] = '/data/agrigorev/02_Projects/ccraft_data'
DEFAULTS['aux_data'] = os.path.join(DEFAULTS['data_root'], 'aux_data')
DEFAULTS['project_dir'] = '/local/home/agrigorev/Workdir/contourcraft_private'
DEFAULTS['experiment_root'] = os.path.join(DEFAULTS['data_root'], 'experiments')

DEFAULTS = munchify(DEFAULTS)
