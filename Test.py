import torch
import matplotlib.pylab as plt
from gan_network import *
from utils import *
import argparse

def parse():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        dest='config', default = 'config_gan_celeba.yml', help='to set the parameters')
    parser.add_argument('--loadfile', default='./experiments/CelebA/GAN/2020-04-19_02-40-02/chkpt/G_epoch_19.pth.tar', type=str,
                        dest='loadfile', help='the path of G model to be loaded')

    return parser.parse_args()

def construct_model(config):

    G = generator(z_size=config.z_size, out_size=config.channel_size, ngf=config.ngf).cuda()
    print('G network structure')
    print(G)
    D = discriminator(in_size=config.channel_size, ndf=config.ndf).cuda()
    print('D network structure')
    print(D)
    return(G, D)

def main():
    torch.manual_seed(1)
    args = parse()
    config = Config(args.config)
    G, _ = construct_model(config)
    # train_loader = get_data(args, config)
    # train_net(G, D, args, config)
    GAN = gan(G, _, config, args, _)
    for _ in range(10):
        noise, image = GAN.generate(args.loadfile,1)

    print('hi')


if __name__=='__main__':
    main()