import torch
import torch.optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import os
import argparse
from gan_network import *
from utils import *
from train import train_net

def parse():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        dest='config', default = 'config_gan_celeba.yml', help='to set the parameters')
    parser.add_argument('--gpu', default=[0], nargs='+', type=int,
                        dest='gpu', help='the gpu used')
    parser.add_argument('--pretrained', default=None,type=str,
                        dest='pretrained', help='the path of pretrained model')
    parser.add_argument('--root', default=None, type=str,
                        dest='root', help='the root of images')
    parser.add_argument('--train_dir', type=str,
                        dest='train_dir', default = './data/img_align_celeba', help='the path of train file')
    parser.add_argument('--save_dir', default='./experiments/CelebA/GAN', type=str,
                        dest='save_dir', help='the path of save generate images')

    return parser.parse_args()


def main():
    torch.manual_seed(1)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    args = parse()
    config = Config(args.config)
    G, D = construct_model(config)
    train_loader = get_data(args, config)
    # for i, (__input, _) in enumerate(train_loader):
    #     print(__input.shape)
    #     plt.imshow(__input[1].permute(1,2,0))
    #     plt.show()
    #     break
    GAN = gan(G, D, config, args, train_loader)
    GAN.train()

    # train_net(G, D, args, config)


if __name__ == '__main__':
    main()
