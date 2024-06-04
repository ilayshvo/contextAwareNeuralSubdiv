import json
import os


def create_experiment():
    exp_name = "Multi-Nets"
    job_path = "./jobs/combined/"
    exp_folder = job_path + "exp/"
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)
    counter = 1
    while os.path.exists(exp_folder + f"{counter:03d}" + "/"):
        counter += 1
    current_folder = exp_folder + f"{counter:03d}" + "/"
    os.makedirs(current_folder)

    diff_out = 128

    data = {
        "wandb_project": "combined",
        "train_pkl": "./data_PKL/fat_dragon_1_train.pkl",
        "valid_pkl": "./data_PKL/fat_dragon_1_train.pkl",
        "output_path": current_folder,
        "exp_name": exp_name + f"{counter:03d}",
        "epochs": 5000,
        "lr": 1e-20,
        "device": 'cuda',
        "Din": diff_out + 6,
        "Dout": diff_out + 6,
        "h_initNet": [128] * 2,
        "h_edgeNet": [128] * 2,
        "h_vertexNet": [128] * 2,
        "use_init": False,
        "numSubd": 4,
        "multi_diff": True,
        "same_nets": False,
        "diff_in": diff_out,
        "diff_out": diff_out,
        "diff_width": 56,
        "diff_dropout": False,
        "diff_blocks": 4,
        "diff_method": "spectral",  # ['spectral', 'implicit_dense']
        "diff_k_eig": 50,  # must be smaller than amount of vertices in lowest level
        "verts_as_feature": False,
        "wandb_log": True
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
        "h_initNet": [32, 32],
        "h_edgeNet": [32, 32],
        "h_vertexNet": [32, 32],
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
