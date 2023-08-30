import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import copy
import time
import os
import torch.optim as optim
from .data_loader import ImageFolder720p


class LocalUpdate(object):
    def __init__(self, args, idxs=None):
        self.args = args
        self.loss_func = nn.MSELoss()
        self.selected_clients = []

        self.ldr_train = DataLoader(
            dataset=ImageFolder720p(args.dataset_path+'D'+str(idxs)),
            batch_size=args.local_bs,
            shuffle=True,
            num_workers=args.num_workers,
        )

    def train(self, net):
#客户端训练
        net.train()
        # train and update
        optimizer = optim.Adam(
            net.parameters(), lr=self.args.lr, weight_decay=1e-5)

        avg_loss, epoch_avg = 0.0, 0.0
        start = time.time()
        for iter in range(self.args.local_ep):
            print(
#输出训练状态
                "Local training epoch [{:3d}/{:3d}]" .format(iter+1, self.args.local_ep))
#转入迭代次数
            for batch_idx, data in enumerate(self.ldr_train, start=1):
                img, patches, _ = data

                patches = patches.to(self.args.device)

                avg_loss_per_image = 0.0
#一次迭代内训练，60个图像块
                for i in range(6):
                    for j in range(10):
                        optimizer.zero_grad()

                        x = patches[:, :, i, j, :, :]
                        y = net(x)
                        loss = self.loss_func(y, x)

                        avg_loss_per_image += (1 / 60) * loss.item()

                        loss.backward()
                        optimizer.step()
                avg_loss += avg_loss_per_image
#综合迭代，输出本地更新的模型
            epoch_avg += avg_loss/len(self.ldr_train)
        end = time.time()
        print('Time consuming of each client {:.1f} s'.format(end - start))
        return net.state_dict(), epoch_avg/self.args.local_ep
