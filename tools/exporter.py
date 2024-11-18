import os
import torch
import numpy as np
from plyfile import PlyData, PlyElement
from omegaconf import OmegaConf

class Gaussian:
    def __init__(self, gaussian, pts_mask):
        self.means = gaussian['_means'][pts_mask].cpu().detach().numpy()
        self.sh_dcs = gaussian['_features_dc'][pts_mask].cpu().detach().numpy()
        self.sh_rests = gaussian['_features_rest'][pts_mask].cpu().detach().numpy()
        self.opacities = gaussian['_opacities'][pts_mask].cpu().detach().numpy()
        self.scales = gaussian['_scales'][pts_mask].cpu().detach().numpy()
        self.quats = gaussian['_quats'][pts_mask].cpu().detach().numpy()

def export_ply(pth_path):
    data = torch.load(pth_path)
#   smpl = SMPLNodes(**cfg.model.SMPLNodes,class_name="SMPLNodes")
#   smpl.load_state_dict(data['models']['SMPLNodes'])
#   gaussian = Gaussian(smpl.get_instance_activated_gs_dict(0))
#    rigid = RigidNodes(**cfg.model.RigidNodes,class_name="RigidNodes")
#    rigid.load_state_dict(data['models']['RigidNodes'])
    for node_type in ['Background', 'RigidNodes', 'DeformableNodes', 'SMPLNodes']:
        gs_dict = data['models'][node_type]
        num_instances = gs_dict["instances_trans"].shape[1] if gs_dict.__contains__('instances_trans') else 1
        for i in range(num_instances):
            pts_mask = gs_dict['points_ids'][..., 0] == i if gs_dict.__contains__('points_ids') else torch.ones(gs_dict['_means'].shape[0], dtype=torch.bool)
            gaussian = Gaussian(gs_dict, pts_mask)
            xyz = gaussian.means
            normals = np.zeros_like(xyz)
            f_dc = gaussian.sh_dcs.reshape((gaussian.sh_dcs.shape[0], -1))
            f_rest = gaussian.sh_rests.reshape((gaussian.sh_rests.shape[0], -1))
            opacities = gaussian.opacities
            scale = gaussian.scales
            rotation = gaussian.quats

            def construct_list_of_attributes(gaussian):
                l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
                # All channels except the 3 DC
                for i in range(3):
                    l.append('f_dc_{}'.format(i))
                for i in range(f_rest.shape[1]):
                    l.append('f_rest_{}'.format(i))
                l.append('opacity')
                for i in range(gaussian.scales.shape[1]):
                    l.append('scale_{}'.format(i))
                for i in range(gaussian.quats.shape[1]):
                    l.append('rot_{}'.format(i))
                return l

            dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(gaussian)]
            attribute_list = [xyz, normals, f_dc, f_rest, opacities, scale, rotation]

            elements = np.empty(xyz.shape[0], dtype=dtype_full)
            attributes = np.concatenate(attribute_list, axis=1)
            # do not save 'features_extra' for ply
            # attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, f_extra), axis=1)
            elements[:] = list(map(tuple, attributes))
            el = PlyElement.describe(elements, 'vertex')
            PlyData([el]).write(node_type + "_" + str(i) + ".ply")

            # save path file
            if gs_dict.__contains__('instances_trans'):
                np.savez(node_type + "_" + str(i) + '_path.npz', trans=gs_dict['instances_trans'].cpu(), quats=gs_dict['instances_quats'].cpu())

if __name__ == '__main__':
    pth_path = 'checkpoint_final.pth'
    export_ply(pth_path)