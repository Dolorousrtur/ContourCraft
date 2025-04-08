import os
import socket

from munch import munchify

hostname = socket.gethostname()

DEFAULTS = dict()

DEFAULTS['project_name'] = 'impulsehood_unimodel'

if hostname == 'ohws68.inf.ethz.ch':
    HOOD_PROJECT = "/local/home/agrigorev/Workdir/00_Projects/contourcraft_private"
    HOOD_DATA = "/local/home/agrigorev/Data/02_Projects/ccraft_data"

    DEFAULTS['server'] = 'stdpc'
    DEFAULTS['CMU_root'] = '/local/home/agrigorev/Data/00_Datasets/AMASS/smpl/CMU'
    DEFAULTS['data_root'] = HOOD_DATA
    DEFAULTS['aux_data'] = os.path.join(HOOD_DATA, 'aux_data')
    DEFAULTS['project_dir'] = HOOD_PROJECT
    DEFAULTS['experiment_root'] = "/media/sdb/Data/experiments/"
elif hostname == 'ait-server-04.inf.ethz.ch':
    HOOD_PROJECT = "/local/home/agrigorev/Workdir/contourcraft_private"
    HOOD_DATA = "/data/agrigorev/02_Projects/ccraft_data"
    DEFAULTS['server'] = 'server4'
    DEFAULTS['CMU_root'] = '/data/agrigorev/00_Datasets/AMASS/smpl/CMU'
    DEFAULTS['data_root'] = HOOD_DATA
    DEFAULTS['aux_data'] = os.path.join(HOOD_DATA, 'aux_data')
    DEFAULTS['project_dir'] = HOOD_PROJECT
    DEFAULTS['experiment_root'] = "/data/agrigorev/experiments/"
    pass
elif hostname.startswith('g'):
    HOOD_PROJECT = "/lustre/home/agrigorev/Workdir/contourcraft_private"
    HOOD_DATA = '/lustre/fast/fast/agrigorev/02_Projects/ccraft_data'
    DEFAULTS['server'] = 'mpi'
    DEFAULTS['CMU_root'] = '/is/cluster/fast/agrigorev/Data/AMASS/smpl/CMU'
    DEFAULTS['data_root'] = HOOD_DATA
    DEFAULTS['aux_data'] = os.path.join(HOOD_DATA, 'aux_data')
    DEFAULTS['project_dir'] = HOOD_PROJECT
    DEFAULTS['experiment_root'] = "/is/cluster/fast/agrigorev/experiments/"
else:
    HOOD_PROJECT = os.environ["HOOD_PROJECT"]
    HOOD_DATA = os.environ["HOOD_DATA"]


    DEFAULTS['server'] = 'local'
    DEFAULTS['data_root'] = HOOD_DATA
    DEFAULTS['experiment_root'] = os.path.join(HOOD_DATA, 'experiments')
    DEFAULTS['aux_data'] = os.path.join(HOOD_DATA, 'aux_data')
    DEFAULTS['project_dir'] = HOOD_PROJECT


DEFAULTS['hostname'] = hostname
DEFAULTS = munchify(DEFAULTS)
