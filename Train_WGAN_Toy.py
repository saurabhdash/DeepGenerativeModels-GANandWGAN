import torch
import torch.optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import os
import argparse
from wgan_network import *
from utils import *
from toy_data import ToyDataset

class FCGenerator(nn.Module):

    def __init__(self, config):
        super(FCGenerator, self).__init__()

        main = nn.Sequential(
            nn.Linear(2, config.DIM),
            nn.ReLU(True),
            nn.Linear(config.DIM, config.DIM),
            nn.ReLU(True),
            nn.Linear(config.DIM, config.DIM),
            nn.ReLU(True),
            nn.Linear(config.DIM, 2),
        )
        self.main = main

    def forward(self, noise):
        noise = noise.squeeze()
        output = self.main(noise)
        return output


class FCDiscriminator(nn.Module):

    def __init__(self, config):
        super(FCDiscriminator, self).__init__()

        main = nn.Sequential(
            nn.Linear(2, config.DIM),
            nn.ReLU(True),
            nn.Linear(config.DIM, config.DIM),
            nn.ReLU(True),
            nn.Linear(config.DIM, config.DIM),
            nn.ReLU(True),
            nn.Linear(config.DIM, 1),
        )
        self.main = main

    def forward(self, inputs):
        output = self.main(inputs)
        return output.view(-1)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def parse():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        dest='config', default = 'config_wgan_toy.yml', help='to set the parameters')
    parser.add_argument('--gpu', default=[0], nargs='+', type=int,
                        dest='gpu', help='the gpu used')
    parser.add_argument('--pretrained', default=None,type=str,
                        dest='pretrained', help='the path of pretrained model')
    parser.add_argument('--root', default=None, type=str,
                        dest='root', help='the root of images')
    parser.add_argument('--train_dir', type=str,
                        dest='train_dir', default = './data', help='the path of train file')
    parser.add_argument('--save_dir', default='./experiments/toy/WGAN', type=str,
                        dest='save_dir', help='the path of save generate images')

    return parser.parse_args()


def main():
    torch.manual_seed(1)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    args = parse()
    config = Config(args.config)
    G = FCGenerator(config).cuda()
    D = FCDiscriminator(config).cuda()
    G.apply(weights_init)
    D.apply(weights_init)
    dset = ToyDataset('swissroll', 10000)
    train_loader = data.DataLoader(dset,  batch_size=config.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    WGAN = wgan_gp(G, D, config, args, train_loader)
    WGAN.train()


if __name__ == '__main__':
    main()
