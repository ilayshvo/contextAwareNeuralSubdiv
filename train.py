from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy
import torch.optim.lr_scheduler

import wandb

from include import *
from models import *
# from torchgptoolbox_nosparse import vertexNormals
# from torchgptoolbox_nosparse.writeOBJ import create_gltf

NETPARAMS = 'netparams.dat'
# torch.set_default_dtype(torch.float64)

# should run the file like "python train.py ./path/to/folder/"
def main():
    folder = sys.argv[1]
    # load hyper parameters
    with open(folder + 'hyperparameters.json', 'r') as f:
        params = json.load(f)
        print(f'Parameters: {params}')

    if params["wandb_log"]:
        wandb.init(
            # set the wandb project where this run will be logged
            project=params["wandb_project"],
            name=params["exp_name"],
            # track hyperparameters and run metadata
            config=params
        )

    # load traininig data
    S = pickle.load(open(params["train_pkl"], "rb"))
    S.computeParameters()
    S.toDevice(params["device"])

    # load validation set - ground truth
    T = pickle.load(open(params["valid_pkl"], "rb"))
    T.computeParameters()
    T.toDevice(params["device"])

    # initialize network
    def init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)

    net = SubdNet(params)
    net = net.to(params['device'])
    net.apply(init_weights)
    net.train()
    # for p in net.parameters():
    #     p.register_hook(lambda grad: torch.clamp(grad, -1, 1))
    total_params = sum(p.numel() for p in net.parameters())
    print(f'#params: {total_params}')
    if params["wandb_log"]:
        wandb.watch(net, log="all")

    # loss function
    lossFunc = torch.nn.MSELoss().to(params['device'])

    # optimizer
    optimizer = torch.optim.Adam(net.parameters())
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 300, 2)

    # training
    trainLossHis = []
    validLossHis = []
    bestLoss = np.inf
    # last_params = {}
    for epoch in range(params['epochs']):
        # if epoch > 0 and epoch % 300 == 0:
        #     for g in optimizer.param_groups:
        #         g['lr'] = g['lr'] * 0.80

        ts = time.time()

        # loop over training shapes
        trainErr = 0.0
        for mIdx in range(S.nM):
            # forward pass
            x_faces = S.getInputData(mIdx)
            outputs = net(x_faces[0], x_faces[1], mIdx, S.hfList, S.poolMats, S.dofs)

            # target mesh
            Vt = S.meshes[mIdx][params["numSubd"]].V.to(params['device'])
            Ft = S.meshes[mIdx][params["numSubd"]].F

            # compute loss function
            loss = 0.0
            for ii in range(params["numSubd"] + 1):
                nV = outputs[ii].size(0)
                loss += lossFunc(outputs[ii], Vt[:nV, :])
                # if math.isnan(loss):
                    # print("nV", nV)
                    # print("ii", ii)
                    # torch.set_printoptions(threshold=100_000)
                    # print("outputs[ii]", outputs[ii])
                    # exit(1)

            # move
            optimizer.zero_grad()
            loss.backward()
            grads_before_clipping = [param.grad.clone() for param in net.parameters() if param.grad is not None]

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

            # Check if any gradient was actually clipped
            gradients_were_clipped = False
            for param, grad_before in zip(net.parameters(), grads_before_clipping):
                if param.grad is not None and not torch.equal(param.grad, grad_before):
                    gradients_were_clipped = True
                    break

            if gradients_were_clipped:
                print("Gradients were clipped!")
            optimizer.step()
            # scheduler.step()

            # for name, param in net.named_parameters():
            #     if epoch != 0:
            #         print(f"{'not ' if torch.equal(param, last_params[name]) else ''}changed: {name}")
            #     last_params[name] = param.detach().clone()
            # record training error
            trainErr += loss.cpu().data.numpy()
        trainLossHis.append(trainErr / S.nM)


        # loop over validation shapes
        validErr = 0.0
        for mIdx in range(T.nM):
            x_faces = T.getInputData(mIdx)
            outputs = net(x_faces[0], x_faces[1], mIdx, T.hfList, T.poolMats, T.dofs)

            # target mesh
            Vt = T.meshes[mIdx][params["numSubd"]].V.to(params['device'])
            Ft = T.meshes[mIdx][params["numSubd"]].F

            # compute loss function
            loss = 0.0
            for ii in range(params["numSubd"] + 1):
                nV = outputs[ii].size(0)
                loss += lossFunc(outputs[ii], Vt[:nV, :])
                # if math.isnan(loss):
                #     print("nV", nV)
                #     print("ii", ii)
                #     torch.set_printoptions(threshold=100_000)
                #     print("outputs[ii]", outputs[ii])
                #     exit(1)

            # record validation error
            validErr += loss.cpu().data.numpy()
        validLossHis.append(validErr / T.nM)

        # save the best model
        if validErr < bestLoss:
            bestLoss = validErr
            torch.save(net.state_dict(), params['output_path'] + NETPARAMS)

        print("epoch %d, train loss %.6e, valid loss %.6e, remain time: %s" % (
            epoch, trainLossHis[-1], validLossHis[-1], int(round((params['epochs'] - epoch) * (time.time() - ts)))))
        if params["wandb_log"]:
            if epoch % 100 == 0:
                # VN = vertexNormals(outputs[-1].cpu(), T.meshes[0][-1].F.to('cpu'))
                # tgp.writeOBJ("temp.obj", outputs[-1].cpu(), T.meshes[0][-1].F.to('cpu'))
                # create_gltf(outputs[-1].cpu(), T.meshes[0][-1].F.to('cpu'))
                wandb.log(
                    {"train loss": trainLossHis[-1], "valid loss": validLossHis[-1], "log loss": np.log10(validLossHis[-1]),
                     "test log loss": np.log10(trainLossHis[-1]),
                     # "gradients": net.named_parameters(),
                     # "lr": scheduler.get_last_lr(),
                     # "top_level_output": wandb.Object3D(open("temp.obj", "r"))
                     })
            else:
                wandb.log(
                    {"train loss": trainLossHis[-1], "valid loss": validLossHis[-1], "log loss": np.log10(validLossHis[-1]),
                     "test log loss": np.log10(trainLossHis[-1]),
                     # "gradients": net.named_parameters()
                     # "lr": scheduler.get_last_lr()
                     })
    # save loss history
    np.savetxt(params['output_path'] + 'train_loss.txt', np.array(trainLossHis), delimiter=',')
    np.savetxt(params['output_path'] + 'valid_loss.txt', np.array(validLossHis), delimiter=',')

    # write output shapes (validation set)
    mIdx = 0
    x_faces = T.getInputData(mIdx)
    net.load_state_dict(torch.load(params['output_path'] + NETPARAMS))
    outputs = net(x_faces[0], x_faces[1], mIdx, T.hfList, T.poolMats, T.dofs)

    # write unrotated outputs
    tgp.writeOBJ(params['output_path'] + str(mIdx) + '_oracle.obj',
                 T.meshes[mIdx][len(outputs) - 1].V.to('cpu'),
                 T.meshes[mIdx][len(outputs) - 1].F.to('cpu'))
    for ii in range(len(outputs)):
        x = outputs[ii].cpu()
        tgp.writeOBJ(params['output_path'] + str(mIdx) + '_subd' + str(ii) + '.obj', x, T.meshes[mIdx][ii].F.to('cpu'))

    wandb.finish()


if __name__ == '__main__':
    main()
