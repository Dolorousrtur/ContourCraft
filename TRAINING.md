# Training a CountourCraft model

Here we describe all the steps necessary for training a ContourCraft model from scratch -- from downloading data to running the training scripts and using the resulting model.

## Data preparation

First, if you haven't already, you'll need to download and set up the path to the auxilliary data folder. To do this, follow [this section in INSTALL.md](INSTALL.md#download-data).

Then, you'll need to download the CMU split of the AMASS dataset that contains SMPL pose sequences we use during the training.

To do this,

1. Go to the [download page of AMASS](https://amass.is.tue.mpg.de/download.php)
2. Find the CMU subset in the table and click on the "SMPL+H G" button to download the gendered version of the SMPL pose sequences.
3. Unpack the downloaded archive to some location and set the variable DEFAULTS.CMU_root to this location in `defaults.py`:
```python
DEFAULTS['CMU_root'] = '/path/to/AMASS/CMU/'
```

Once the data preparation steps are finished, you may proceed to run the training script.

## Training script

The training procedure for ContourCraft consists of two stages:

1. First, we pretrain a model with a standard set of physical losses and a repulsion loss, so that it learns to generate realistic cloth dynamics and *prevent* cloth intersections.
2. Then, we finetune this model with the Contour Loss to teach it to *resolve* intersections.

These two stages use two separate configuration files and need to be run sequentially.

### Stage 1
To run the first stage, run the training script with the configuration file `ccraft_train/stage1`:

```bash
python train.py config=ccraft_train/stage1
```

The training will run for 56000 training iterations and will produce checkpoints under `DEFAULTS.data_root/trained_models/new_training/stage1/`

### Stage 2
Once you have a checkpoint produced by the first stage, you can start the second one, which uses the configuration file `ccraft_train/stage2`

Before running it, make sure that the path `restart.checkpoint_path` in `configs/ccraft_train/stage2.yaml` leads to the correct checkpoint path under `DEFAULTS.data_root`:
```yaml
restart:
  checkpoint_path: "trained_models/new_training/stage1/step_0000056000.pth"
  step_start: 45000
```

Then, run the second stage with:
```bash
python train.py config=ccraft_train/stage2
```
This training will start from the iteration 45000 until iteration 56000. The final checkpoint will be stored under `DEFAULTS.data_root/trained_models/new_training/stage2/step_0000056000.pth`

## Inference 

To run the simulation with the newly trained checkpoint, refer to the notebooks `Inference.ipynb` and `Inference_from_mesh_sequence.ipynb` and set the `checkpoint_path` there to the new checkpoint:

```python
checkpoint_path = Path(DEFAULTS.data_root) / 'trained_models' / 'new_training' / 'stage2' / 'step_0000056000.pth'
```