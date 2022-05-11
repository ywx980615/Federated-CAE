import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
from matplotlib import pyplot as plt
import copy
import time
import os


from src.models.cae_32x32x32_zero_pad_bin import CAE
from src.logger import Logger
from src.namespace import Namespace
from src.utils import save_imgs
from src.data_loader import ImageFolder720p
from src.config import args_parser
from src.Client import LocalUpdate
import argparse
from pathlib import Path


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


logger = Logger(__name__, colorize=True)


def train(args):

    root_dir = Path(__file__).resolve().parents[0]

    logger.info("training: experiment %s" % ('training'))
    exp_dir = root_dir / "experiments_FL" / 'training'
    for d in ["out", "checkpoint", "logs"]:
        os.makedirs(exp_dir / d, exist_ok=True)

    net_glob = CAE()
    w_glob = net_glob.state_dict()
    loss_train = []

    for epoch_idx in range(1, args.epochs+1):
        print('----Epoch------', epoch_idx)
        start = time.time()
        loss_locals = []
        w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for i, idx in enumerate(idxs_users):
            print("Client  [{:3d}/{:3d}]" .format(i+1, m))
            local = LocalUpdate(args=args,  idxs=idx)
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))

            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob and save
        net_glob.load_state_dict(w_glob)
        torch.save(net_glob.state_dict(), exp_dir /
                   f"checkpoint/model_{epoch_idx}.pth")

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)
        print('Round {:3d}, Average loss {:.3f}'.format(epoch_idx, loss_avg))

        end = time.time()
        print('Time consuming for a global epoch {:.1f} s'.format(end - start))
    save_data = {'loss_train': loss_train}
    torch.save(save_data, exp_dir / "loss_train")
    torch.save(net_glob.state_dict(), exp_dir / "model_final.pth")


if __name__ == "__main__":
    args = args_parser()
    args.device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')

    train(args)
