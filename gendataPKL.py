from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from include import *


def main():
    # validation PKL
    mesh_folders = ['./data_meshes/coseg_aliens_30/']
    # mesh_folders = ['./data_meshes/bunny/', './data_meshes/rockerArm/', './data_meshes/fertility/']
    S = TrainMeshes(mesh_folders)

    pickle.dump(S, file=open("./data_PKL/coseg_aliens_30_valid.pkl", "wb"))

    # training PKL
    mesh_folders = ['./data_meshes/coseg_aliens_30_test/']
    # mesh_folders = ['./data_meshes/bunny/', './data_meshes/rockerArm/', './data_meshes/fertility/']
    S = TrainMeshes(mesh_folders)

    pickle.dump(S, file=open("./data_PKL/coseg_aliens_30_train.pkl", "wb"))


if __name__ == '__main__':
    main()
