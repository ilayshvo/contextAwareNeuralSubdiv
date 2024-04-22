import json
import os

def create_experiment():
    exp_name = "hpc:"
    job_path = "./jobs/man_head_1/"
    exp_folder = job_path + "exp/"
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)
    counter = 1
    while os.path.exists(exp_folder + f"{counter:03d}" + "/"):
        counter += 1
    current_folder = exp_folder + f"{counter:03d}" + "/"
    os.makedirs(current_folder)

    diff_out = 61

    data = {
        "wandb_project": "L2P-CANSubdiv-man-head",
        "train_pkl": "./data_PKL/man_head_1_norm_train.pkl",
        "valid_pkl": "./data_PKL/man_head_1_norm_valid.pkl",
        "data_path": "./data_meshes/man_head_1",
        "output_path": current_folder,
        "exp_name": exp_name + f"{counter:03d}",
        "epochs": 15000,
        "lr": 2e-3,
        "device": 'cuda',
        "Din": diff_out + 6,
        "Dout": 31,
        "h_initNet": [256] * 10,
        "h_edgeNet": [256] * 10,
        "h_vertexNet": [256] * 10,
        "use_init": True,
        "numSubd": 2,
        "diff_in": 16,
        "diff_out": diff_out,
        "diff_width": 56,
        "diff_dropout": False,
        "diff_blocks": 4,
        "diff_method": "spectral",  # ['spectral', 'implicit_dense']
        "l2p_out": diff_out
    }

    # write hyper parameters into a json file
    with open(data['output_path'] + 'hyperparameters.json', 'w') as f:
        json.dump(data, f, indent=4)

    print(data['output_path'])
def main():

    data = {
        "train_pkl": "./data_PKL/cartoon_elephant_train.pkl",
        "valid_pkl": "./data_PKL/cartoon_elephant_valid.pkl",
        "output_path": './jobs/net_cartoon_elephant/',
        "epochs": 700, 
        "lr": 2e-3, 
        "device": 'cuda',
        "Din": 6,
        "Dout": 32,
        "h_initNet": [32,32],
        "h_edgeNet": [32,32],
        "h_vertexNet": [32,32],
        "numSubd": 2,
    }

    # create directory
    if not os.path.exists(data['output_path']):
        os.mkdir(data['output_path'])

    # write hyper parameters into a json file
    with open(data['output_path'] + 'hyperparameters.json', 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':
    # main()
    create_experiment()