import gc
import os
import copy

import numpy as np
import trimesh

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class L2PDataset(Dataset):
    def __init__(self, args, set_type='train', is_train=True):
        if not os.path.exists(args["data_path"]):
            print("Error")

        # init
        self.args = args
        self.set_type = set_type
        self.is_train = is_train

        self._load_dataset()

    def _load_dataset(self):
        datas = list()

        max_vertex_num = -np.inf

        subd_level = "subd0"
        # data path
        mesh_path = os.path.join(self.args["data_path"], self.set_type + '_norm', subd_level)
        vector_path = os.path.join(self.args["data_path"], 'eigen_vectors', self.set_type, subd_level)
        # eigen_values_path = os.path.join(self.args["data_path"], 'eigen_values', self.set_type)
        hks_path = os.path.join(self.args["data_path"], 'HKS', self.set_type, subd_level)
        gaussian_curvature_path = os.path.join(self.args["data_path"], 'gaussian_curvatures', self.set_type, subd_level)
        V_dihedral_angles_path = os.path.join(self.args["data_path"], 'V_dihedral_angles', self.set_type, subd_level)

        if self.is_train == False:
            vf_list = list()

            max_vertex_negibor_face_num = -np.inf

        file_list = sorted(os.listdir(mesh_path))

        # load the size of eigen_vectors is max_input, this operator can save memory.
        max_input = np.max(self.args["num_inputs"])

        for file in file_list:
            file.strip()
            if os.path.isdir(os.path.join(mesh_path, file)):
                continue
            print(os.path.join(mesh_path, file))

            # load mesh
            mesh = trimesh.load(os.path.join(mesh_path, file), process=False, force='mesh')
            mesh: trimesh.Trimesh

            eigen_vector = torch.from_numpy(
                np.load(os.path.join(vector_path, os.path.splitext(file)[0] + '_eigen.npy'))[:, :max_input]).float()

            hks_cat = torch.from_numpy(
                np.load(os.path.join(hks_path, os.path.splitext(file)[0] + '_hks.npy'))).float()

            # normalize the gaussian curvature
            gaussian_curvature = torch.exp(-(torch.from_numpy(np.load(os.path.join(gaussian_curvature_path,
                                                                                   os.path.splitext(file)[
                                                                                       0] + '_gaussian_curvature.npy'))).float()))
            gaussian_curvature = ((gaussian_curvature - gaussian_curvature.min()) / (
                    gaussian_curvature.max() - gaussian_curvature.min())).unsqueeze(1)

            V_dihedral_angles = torch.from_numpy(
                np.load(os.path.join(V_dihedral_angles_path, os.path.splitext(file)[0] + '_V_dihedralAngles.npy')))

            # the input features
            features = torch.from_numpy(np.concatenate(
                (mesh.vertices, mesh.vertex_normals, eigen_vector[:, 1:21], gaussian_curvature, V_dihedral_angles, hks_cat), axis=1))

            max_vertex_num = np.maximum(max_vertex_num, mesh.vertices.shape[0])

            eigen_list = list()
            levels = list()
            final_mat = None
            for i, k in enumerate(self.args["num_inputs"]):
                level = eigen_vector[:, :k].T.to(self.args["device"]) @ features.float().to(self.args["device"])
                levels.append(level.cpu())
                eigen_list.append(eigen_vector[:, :k].cpu())
                if i == 0:
                    final_mat = eigen_vector[:, :k]

            # different level
            c_list = list()
            c_list.append((eigen_list[1].T @ eigen_list[0]).float())
            c_list.append((eigen_list[2].T @ eigen_list[1]).float())
            c_list.append((eigen_list[0].T @ eigen_list[2]).float())

            data = dict()
            data['levels'] = levels
            data['c'] = c_list
            data['final_mat'] = final_mat
            data['mesh_path'] = os.path.join(mesh_path, file)
            datas.append(data)

            if self.is_train == False:
                max_vertex_negibor_face_num = np.maximum(max_vertex_negibor_face_num, mesh.vertex_faces.shape[1])

                vf_list.append(torch.from_numpy(copy.copy(mesh.vertex_faces)).long())

            del mesh, eigen_vector, hks_cat, gaussian_curvature, V_dihedral_angles, features, eigen_list, levels, c_list, data, final_mat
            gc.collect()

        for i, d in enumerate(datas):
            if (int(max_vertex_num - d['final_mat'].shape[0]) != 0):
                temp = d['final_mat'].transpose(0, 1)
                d['final_mat'] = F.pad(temp, pad=(0, int(max_vertex_num - d['final_mat'].shape[0]))).transpose(0, 1)

        self.datas = datas

        # the data for test mode
        if self.is_train == False:
            # padding vf_list
            for i, f in enumerate(vf_list):
                if (int(max_vertex_negibor_face_num - f.shape[1]) != 0) or (int(max_vertex_num - f.shape[0]) != 0):
                    vf_list[i] = F.pad(f, pad=(
                        0, int(max_vertex_negibor_face_num - f.shape[1]), 0, int(max_vertex_num - f.shape[0])),
                                       value=-1)

            self.vf = vf_list

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        if self.set_type == 'train':
            return self.datas[item]
        else:
            return self.datas[item], self.vf[item]
