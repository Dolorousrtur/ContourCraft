import torch

from utils import lbs as my_lbs


class GarmentSMPL:
    """
    Generates garment geometry in the given poses using SMPL blend shapes and linear blend skinning.
    """
    def __init__(self, body_model, garment_skinning_dict):
        self.smpl_model = body_model
        self.garment_skinning_dict = garment_skinning_dict

    def make_vertices(self, betas: torch.FloatTensor, full_pose: torch.FloatTensor,
                      transl: torch.FloatTensor = None) -> torch.FloatTensor:
        """
        Generates garment vertices for the given pose and shape parameters.
        :param betas: [Nx10] shape parameters
        :param full_pose: [Nx72] pose parameters in axis-angle format
        :param transl: [Nx3] grobal translation
        :return: [NxVx3] garment vertices
        """

        n_joints_body = full_pose.shape[1] // 3
        n_joints_garment = self.garment_skinning_dict['posedirs'].shape[0] // 9 + 1

        if n_joints_body != n_joints_garment:
            raise ValueError(f"The pose sequence and the used garment template have different numbers of pose joints. "
                             f"({n_joints_body} and {n_joints_garment} respectively) "
                             "Make sure, both pose sequence and the garment template use the same body model (e.g. SMPL or SMPLX).")

        J_transformed, joint_transforms = my_lbs.get_transformed_joints(betas, full_pose, self.smpl_model.v_template,
                                                                        self.smpl_model.shapedirs,
                                                                        self.smpl_model.J_regressor,
                                                                        self.smpl_model.parents)
        

        v_garment, _ = my_lbs.pose_garment(betas, full_pose, self.garment_skinning_dict['v'],
                                           self.garment_skinning_dict['shapedirs'],
                                           self.garment_skinning_dict['posedirs'],
                                           self.garment_skinning_dict['lbs_weights'], 
                                           J_transformed, joint_transforms)

        if transl is not None:
            v_garment = v_garment + transl[:, None]

        return v_garment
