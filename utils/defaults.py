import os
import socket

from munch import munchify

hostname = socket.gethostname()

HOOD_PROJECT = "/home/agrigorev/Workdir/00_Projects/prj4_sandbox/ccraft"
HOOD_DATA = "/home/agrigorev/Data/01_Projects/hood_public"

if "HOOD_PROJECT" in os.environ:
    HOOD_PROJECT = os.environ["HOOD_PROJECT"]
    HOOD_DATA = os.environ["HOOD_DATA"]

os.environ["HOOD_PROJECT"] = HOOD_PROJECT
os.environ["HOOD_DATA"] = HOOD_DATA

DEFAULTS = dict()

DEFAULTS['server'] = 'local'
DEFAULTS['data_root'] = HOOD_DATA
DEFAULTS['experiment_root'] = os.path.join(HOOD_DATA, 'experiments')
DEFAULTS['vto_root'] = os.path.join(HOOD_DATA, 'vto_dataset')
DEFAULTS['aux_data'] = os.path.join(HOOD_DATA, 'aux_data')
DEFAULTS['project_dir'] = HOOD_PROJECT


DEFAULTS['hostname'] = hostname
DEFAULTS = munchify(DEFAULTS)
