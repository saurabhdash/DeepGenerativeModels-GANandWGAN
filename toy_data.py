import torch
import numpy as np
import scipy
from torch.utils import data
import sys
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets
import random


class ToyDataset(data.Dataset):

    def __init__(self, distribution, N):
        self.distribution = distribution
        self.N = N
        if distribution == 'gaussian25':
            self.data = self.gaussian25()
        elif distribution == 'swissroll':
            self.data = self.swissroll()
        elif distribution == 'gaussian8':
            self.data = self.gaussian8()
        else:
            NotImplementedError()

    def gaussian25(self):
        dataset = []
        for i in np.arange(1, self.N,1):
            for x in range(-2, 3):
                for y in range(-2, 3):
                    point = np.random.randn(2) * 0.05
                    point[0] += 2 * x
                    point[1] += 2 * y
                    dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        dataset /= 2.828  # stdev
        return dataset

    def swissroll(self):
        data = sklearn.datasets.make_swiss_roll(
            n_samples=self.N,
            noise=0.25
        )[0]
        data = data.astype('float32')[:, [0, 2]]
        data /= 7.5  # stdev plus a little
        return data

    def gaussian8(self):

        scale = 2.
        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1. / np.sqrt(2), 1. / np.sqrt(2)),
            (1. / np.sqrt(2), -1. / np.sqrt(2)),
            (-1. / np.sqrt(2), 1. / np.sqrt(2)),
            (-1. / np.sqrt(2), -1. / np.sqrt(2))
        ]
        centers = [(scale * x, scale * y) for x, y in centers]
        dataset = []

        for i in range(self.N):
            point = np.random.randn(2) * .02
            center = random.choice(centers)
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)

        dataset = np.array(dataset, dtype='float32')
        dataset /= 1.414  # stdev
        return dataset



    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return (self.data[idx], torch.tensor(0))


# dset = ToyDataset('gaussian8', 10000)
# dloader = data.DataLoader(dset,  batch_size=1000, shuffle=True, num_workers=1, pin_memory=True)
#
# for i, val in enumerate(dloader):
#     print(val)
#     plt.scatter(val[:,0], val[:,1])
#     plt.show()
#     break
#
# print('hi')
