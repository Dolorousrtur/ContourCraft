from collections import defaultdict
from pathlib import Path
from utils.datasets import build_smpl_bygender
from utils.defaults import DEFAULTS
from utils.io import load_obj, pickle_dump, pickle_load
import numpy as np
import pandas as pd
import yaml

from utils.mesh_creation import GarmentCreator


class GauGarConverter:
    def __init__(self, gaugar_data_root, gaugar_output_root, body_models_root, garment_dicts_dir, checkpoint_path):
        self.gaugar_data_root = gaugar_data_root
        self.gaugar_output_root = gaugar_output_root
        self.body_models_root = body_models_root
        self.body_models_dict = build_smpl_bygender(body_models_root, 'smplx', use_pca=True)
        self.garment_dicts_dir = garment_dicts_dir
        self.checkpoint_path = checkpoint_path


    def _convert_garment_dir(self, registrations_dir, out_file_path):

        garment_files = sorted(list(registrations_dir.glob("*.obj")))

        faces = None
        vertices_list = []

        uv_faces = None
        uv_vertices_list = []

        for garment_file in garment_files:
            vertices, faces, uvs, uv_faces = load_obj(garment_file, tex_coords=True)
            vertices_list.append(vertices)
            uv_vertices_list.append(uvs)

        vertices = np.stack(vertices_list, axis=0)
        uv_vertices = np.stack(uv_vertices_list, axis=0)

        out_dict = dict()
        out_dict['vertices'] = vertices
        out_dict['faces'] = faces
        out_dict['uv_coords'] = uv_vertices
        out_dict['uv_faces'] = uv_faces


        out_dict['pred'] = vertices
        out_dict['cloth_faces'] = faces

        out_file_path.parent.mkdir(parents=True, exist_ok=True)
        pickle_dump(out_dict, out_file_path)

    def _convert_smplx_dir(self, smplx_dir, out_file_path, gender):
        smplx_files = sorted(list(smplx_dir.glob("*.pkl")))

        smplx_params_sequence = defaultdict(list)
        for smplx_file in smplx_files:
            smplx_params = pickle_load(smplx_file)
            for k, v in smplx_params.items():
                smplx_params_sequence[k].append(v)

        for k, v in smplx_params_sequence.items():
            smplx_params_sequence[k] = np.stack(v, axis=0)

        smplx_model = self.body_models_dict[gender]

        left_hand_pose = smplx_params_sequence['left_hand_pose'][:, :6]
        left_hand_pose = left_hand_pose @ smplx_model.left_hand_components.cpu().numpy()

        right_hand_pose = smplx_params_sequence['right_hand_pose'][:, :6]
        right_hand_pose = right_hand_pose @ smplx_model.right_hand_components.cpu().numpy()

        cmu_format_sequence = dict()
        cmu_format_sequence['betas'] = smplx_params_sequence['betas']
        cmu_format_sequence['expression'] = smplx_params_sequence['expression']
        cmu_format_sequence['trans'] = smplx_params_sequence['transl']
        cmu_format_sequence['root_orient'] = smplx_params_sequence['global_orient']
        cmu_format_sequence['pose_body'] = smplx_params_sequence['body_pose']
        cmu_format_sequence['pose_hand'] = np.concatenate([left_hand_pose, 
                                                            right_hand_pose], axis=-1)
        cmu_format_sequence['pose_jaw'] = smplx_params_sequence['jaw_pose']
        cmu_format_sequence['pose_eye'] = np.concatenate([smplx_params_sequence['leye_pose'],
                                                            smplx_params_sequence['reye_pose']], axis=-1)


        cmu_format_sequence['mocap_frame_rate'] = 30

        np.savez_compressed(out_file_path, **cmu_format_sequence)

    def _convert_garment_subject(self, subject_id):
        print(f"Converting registered garment sequences...")
        subject_dir = self.gaugar_output_root / subject_id
        stage2_dir = subject_dir / 'stage2'
        sequence_dirs = sorted(list(stage2_dir.iterdir()))
        for sequence_dir in sequence_dirs:
            sequence_name = sequence_dir.name

            registrations_dir = sequence_dir / 'meshes'

            out_path = subject_dir / 'stage4' / 'registrations' / f"{sequence_name}.pkl"
            print(f"Converting {sequence_name} to {out_path}...")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            self._convert_garment_dir(registrations_dir, out_path)

        print('Done!')

    def _convert_smplx_subject(self, subject_id, subject_out, gender):
        print(f"Converting SMPLX sequences...")
        subject_dir = self.gaugar_data_root / subject_id
        subject_out_dir = self.gaugar_output_root / subject_out
        sequence_dirs = sorted(list(subject_dir.iterdir()))
        for sequence_dir in sequence_dirs:
            sequence_name = sequence_dir.name

            smplx_dir = sequence_dir / 'smplx'
            out_path = subject_out_dir / 'stage4' / 'smplx' / f"{sequence_name}.npz"
            print(f"Converting {sequence_name} to {out_path}...")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            self._convert_smplx_dir(smplx_dir, out_path, gender)

        print('Done!')

    def _create_datasplit(self, smplx_files, subject_out, gender):

        datasplit = defaultdict(list)
        for smplx_file in smplx_files:
            smplx_params = dict(np.load(smplx_file))
            subject_id = smplx_file.stem
            datasplit['id'].append(subject_id)
            datasplit['length'].append(smplx_params['pose_body'].shape[0])
            datasplit['garment'].append(subject_out)
            datasplit['gender'].append(gender)

        datasplit = pd.DataFrame(datasplit)
        return datasplit

    def _create_datasplits(self, subject_out, gender, n_validation=1):
        stage4_dir = self.gaugar_output_root / subject_out / 'stage4'
        smplx_dir = stage4_dir / 'smplx'

        #id, length, garment, gender
        smplx_files = sorted(list(smplx_dir.glob("*.npz")))

        valid_files = smplx_files[:n_validation]
        train_files = smplx_files[n_validation:]

        valid_datasplit = self._create_datasplit(valid_files, subject_out, gender)
        train_datasplit = self._create_datasplit(train_files, subject_out, gender)

        out_datasplit_dir = Path(DEFAULTS.aux_data) / 'datasplits' / 'finetuning' / subject_out
        out_datasplit_dir.mkdir(parents=True, exist_ok=True)
        valid_datasplit.to_csv(out_datasplit_dir / 'valid.csv', index=False)
        train_datasplit.to_csv(out_datasplit_dir / 'train.csv', index=False)
        print(f"Datasplits saved to {out_datasplit_dir}")


    def _convert_smplx_dict_for_import(self, smplx_dict):
        converted_dict = dict()
        # 'betas', 'transl', 'global_orient', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'left_hand_pose', 'right_hand_pose'
        converted_dict['betas'] = smplx_dict['betas']
        converted_dict['transl'] = smplx_dict['trans']
        converted_dict['global_orient'] = smplx_dict['root_orient']
        converted_dict['body_pose'] = smplx_dict['pose_body']
        converted_dict['jaw_pose'] = smplx_dict['pose_jaw']
        converted_dict['leye_pose'] = smplx_dict['pose_eye'][:, :3]
        converted_dict['reye_pose'] = smplx_dict['pose_eye'][:, 3:]
        converted_dict['left_hand_pose'] = smplx_dict['pose_hand'][:, :45]
        converted_dict['right_hand_pose'] = smplx_dict['pose_hand'][:, 45:]

        return converted_dict

    def _import_garment(self, subject_out, template_sequence, template_frame, gender, pinned_indices=None):
        gc = GarmentCreator(self.garment_dicts_dir, self.body_models_root, 'smplx', gender, 
                    n_samples_lbs=0, verbose=True, coarse=True, approximate_center=True, swap_axes=False)

        subject_dir = self.gaugar_output_root / subject_out
        template_obj_path = subject_dir / 'stage1' / 'template_uv.obj'
        obj_dict = gc._load_from_obj(template_obj_path)

        smplx_sequence_path = subject_dir / 'stage4' / 'smplx' / f"{template_sequence}.npz"
        smplx_params = dict(np.load(smplx_sequence_path))
        smplx_params = {k: v[template_frame:template_frame+1] for k, v in smplx_params.items() if k != 'mocap_frame_rate'}
        smplx_params = self._convert_smplx_dict_for_import(smplx_params)

        relaxation_trajectory = gc._add_posed_garment_raw(obj_dict, 
                                  subject_out, 
                                  smplx_params, 
                                  self.checkpoint_path, 
                                  n_relaxation_steps=30, 
                                  pinned_indices=pinned_indices,
                                  gender=gender)

        return relaxation_trajectory
    
    def _create_config(self, subject_out, multigarment=False):
        template_config_path = Path(DEFAULTS.project_dir) / 'configs' / 'finetune' / 'base.yaml'
        config = yaml.safe_load(template_config_path.read_text())
        config['dataloaders']['finetune']['dataset']['finetune']['train_split_path'] = \
             str(Path('datasplits') / 'finetuning' / subject_out / 'train.csv')
        config['dataloaders']['finetune']['dataset']['finetune']['valid_split_path'] = \
             str(Path('datasplits') / 'finetuning' / subject_out / 'valid.csv'        )

        if multigarment:
            config['dataloaders']['finetune']['dataset']['finetune'].pop('registration_root')
            config['dataloaders']['finetune']['dataset']['finetune'].pop('body_sequence_root')
        else:
            config['dataloaders']['finetune']['dataset']['finetune']['registration_root'] = \
                    str(Path(self.gaugar_output_root) / subject_out / 'stage4' / 'registrations')
            config['dataloaders']['finetune']['dataset']['finetune']['body_sequence_root'] = \
                    str(Path(self.gaugar_output_root) / subject_out / 'stage4' / 'smplx')
        

        config['checkpoints_dir'] = f"trained_models/finetuning/{subject_out}"
        
        out_config_path = Path(DEFAULTS.project_dir) / 'configs' / 'finetune' / f'{subject_out}.yaml'
        out_config_path.parent.mkdir(parents=True, exist_ok=True)
        out_config_path.write_text(yaml.dump(config, default_flow_style=False))
        relative_config_path = out_config_path.relative_to(Path(DEFAULTS.project_dir) / 'configs').with_suffix('')

        print(f"Config saved to {out_config_path}")
        print('Now you can finetune the model with:')
        print(f"python train.py config={relative_config_path}")
        return out_config_path


    def convert_subject(self, subject, gender, template_sequence, template_frame, subject_out=None):
        subject_out = subject_out or subject
        self._convert_smplx_subject(subject, subject_out, gender)
        self._convert_garment_subject(subject_out)
        self._import_garment(subject_out, template_sequence, template_frame, gender)
        self._create_datasplits(subject_out, gender)
        self._create_config(subject_out)

    def _convert_df_multigarment(self, df, subject_name):
        n_items = len(df)
        registration_root = Path(self.gaugar_output_root) / subject_name / 'stage4' / 'registrations'
        smplx_root = Path(self.gaugar_output_root) / subject_name / 'stage4' / 'smplx'
        df['registration_root'] = [str(registration_root)] * n_items
        df['body_sequence_root'] = [str(smplx_root)] * n_items
        return df

    def _create_datasplits_multigarment(self, subject_list, experiment_name):
        train_datasplit_list = []
        valid_datasplit_list = []

        for i, subject_out in enumerate(subject_list):
            train_df_path = Path(DEFAULTS.aux_data) / 'datasplits' / 'finetuning' / subject_out / 'train.csv'
            valid_df_path = Path(DEFAULTS.aux_data) / 'datasplits' / 'finetuning' / subject_out / 'valid.csv'
            train_df = pd.read_csv(train_df_path)
            valid_df = pd.read_csv(valid_df_path)

            train_df = self._convert_df_multigarment(train_df, subject_out)
            valid_df = self._convert_df_multigarment(valid_df, subject_out)

            train_datasplit_list.append(train_df)
            valid_datasplit_list.append(valid_df)

        train_datasplit = pd.concat(train_datasplit_list, axis=0)
        valid_datasplit = pd.concat(valid_datasplit_list, axis=0)

        out_datasplit_dir = Path(DEFAULTS.aux_data) / 'datasplits' / 'finetuning' / experiment_name
        out_datasplit_dir.mkdir(parents=True, exist_ok=True)

        train_datasplit.to_csv(out_datasplit_dir / 'train.csv', index=False)
        valid_datasplit.to_csv(out_datasplit_dir / 'valid.csv', index=False)
        print(f"Datasplits saved to {out_datasplit_dir}")

    def prepare_multigarment(self, subject_list, experiment_name):
        self._create_datasplits_multigarment(subject_list, experiment_name)
        self._create_config(experiment_name, multigarment=True)
            