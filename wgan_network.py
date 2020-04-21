import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils import *
from torch.autograd import Variable, grad
from datetime import datetime
import torchvision.utils as vutils
from torchvision import datasets
import io
import PIL.Image
from torchvision.transforms import ToTensor

# G(z)
class generator_noBN(nn.Module):
    '''
        Generative Network
    '''
    def __init__(self, z_size=100, out_size=3, ngf=128):
        super(generator_noBN, self).__init__()
        self.z_size = z_size
        self.ngf = ngf
        self.out_size = out_size

        self.main = nn.Sequential(
            # input size is z_size
            nn.ConvTranspose2d(self.z_size, self.ngf * 8, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(inplace=True),
            # state size: (ngf * 8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(inplace=True),
            # state size: (ngf * 4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(inplace=True),
            # state size: (ngf * 2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(self.ngf),
            nn.ReLU(inplace=True),
            # state size: ngf x 32 x 32
            nn.ConvTranspose2d(self.ngf, self.out_size, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size: out_size x 64 x 64
        )

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input):
        output = self.main(input)
        return output

# D
class discriminator_noBN(nn.Module):
    '''
        Critic Network
    '''
    def __init__(self, in_size=3, ndf=128):
        super(discriminator_noBN, self).__init__()
        self.in_size = in_size
        self.ndf = ndf

        self.main = nn.Sequential(
            # input size is in_size x 64 x 64
            nn.Conv2d(self.in_size, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: ndf x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf * 2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf * 4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf * 8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
            # state size: 1 x 1 x 1
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input):
        output = self.main(input)
        return output


class wgan_gp(nn.Module):
    '''
        Class that defines a Generative Adversarial Interaction between 2 networks.
        Loss is Wasserstein loss and 1-lipschitzness is maintained using gradient penalty.
    '''
    def __init__(self, G, D, config, args, train_loader):
        super(wgan_gp, self).__init__()
        self.G = nn.DataParallel(G)
        print('G network structure')
        print(self.G)
        self.D = nn.DataParallel(D)
        print('D network structure')
        print(self.D)
        self.config = config
        self.args = args
        self.train_loader = train_loader
        self.writer = SummaryWriter()

    def train(self):

        self.init_train()

        for epoch in range(self.config.epoches):
            self.epoch = epoch

            for i, (__input, _) in enumerate(self.train_loader):
                self.data_time.update(time.time() - self.start)

                #################################
                #       Update Critic           #
                #################################
                for params in self.D.parameters():  # reset requires_grad
                    params.requires_grad = True  # they are set to False below in netG update

                for critic_iter in range(self.config.critic_iters):

                    batch_size = __input.size(0)
                    input_var = __input.cuda()
                    self.D.zero_grad()
                    # # Train discriminator with real data
                    # label_real = torch.ones(batch_size)
                    # label_real_var = Variable(label_real.cuda())
                    # print(input_var.is_cuda)
                    # print(next(self.D.module.parameters()).is_cuda)
                    D_real_result = self.D(input_var)
                    D_real_loss = D_real_result.mean()

                    D_real_loss.backward(self.mone)

                    noise_var = torch.randn((batch_size, self.config.z_size)).view(-1, self.config.z_size, 1, 1).to('cuda')
                    # print(noise_var.shape)
                    # print(noise_var.squeeze().shape)
                    G_result = self.G(noise_var).clone()

                    D_fake_result = self.D(G_result).squeeze()
                    D_fake_loss = D_fake_result.mean()
                    D_fake_loss.backward(self.one)

                    gradient_penalty = self.calc_gradient_penalty(input_var.data, G_result.data, batch_size)
                    gradient_penalty.backward()

                    D_train_loss = D_fake_loss - D_real_loss + gradient_penalty
                    self.D_losses.update(D_train_loss.item())
                    self.Wasserstein_D.update((D_real_loss - D_fake_loss).item())

                    self.optimizerD.step()


                #################################
                #       Update Generator        #
                #################################

                for p in self.D.parameters():
                    p.requires_grad = False  # to avoid computation

                self.G.zero_grad()

                noise = torch.randn((batch_size, self.config.z_size)).view(-1, self.config.z_size, 1, 1)
                noise_var = noise.cuda()
                # print(noise_var.shape)
                G_result = self.G(noise_var)

                D_fake_result = self.D(G_result).squeeze()
                D_fake_loss = D_fake_result.mean()
                D_fake_loss.backward(self.mone)
                self.G_losses.update(-D_fake_loss.item())
                self.optimizerG.step()

                self.batch_time.update(time.time() - self.start)
                self.start = time.time()

                if (i + 1) % self.config.display == 0:
                    print_log(epoch + 1, self.config.epoches, i + 1, len(self.train_loader), self.config.base_lr,
                              self.config.display, self.batch_time, self.data_time, self.D_losses, self.G_losses)
                    self.writer.add_scalar('Loss/D', self.D_losses.val, self.iters)
                    self.writer.add_scalar('Loss/G', self.G_losses.val, self.iters)
                    self.writer.add_scalar('Loss/Wasserstein Distance', self.Wasserstein_D.val, self.iters)

                    with torch.no_grad():
                        fake = self.G(self.fixed_noise).detach().cpu()

                    if self.config.dataset == 'toy':
                        self.data_dist(input_var)
                    else:
                        grid = vutils.make_grid(fake, padding=2, normalize=True)
                        self.writer.add_image('Generated Examples', grid, self.iters)

                    self.batch_time.reset()
                    self.data_time.reset()
                elif (i + 1) == len(self.train_loader):
                    print_log(epoch + 1, self.config.epoches, i + 1, len(self.train_loader), self.config.base_lr,
                              (i + 1) % self.config.display, self.batch_time, self.data_time, self.D_losses, self.G_losses)
                    self.writer.add_scalar('Loss/D', self.D_losses.val, self.iters)
                    self.writer.add_scalar('Loss/G', self.G_losses.val, self.iters)

                    with torch.no_grad():
                        fake = self.G(self.fixed_noise).detach().cpu()

                    if self.config.dataset == 'toy':
                        self.data_dist(input_var)
                    else:
                        grid = vutils.make_grid(fake, padding=2, normalize=True)
                        self.writer.add_image('Generated Examples', grid, self.iters)

                    self.batch_time.reset()
                    self.data_time.reset()

                self.iters += 1

            self.D_loss_list.append(self.D_losses.avg)
            self.G_loss_list.append(self.G_losses.avg)

            self.writer.add_scalar('Epoch_Loss/D', self.D_losses.avg, self.epoch)
            self.writer.add_scalar('Epoch_Loss/G', self.G_losses.avg, self.epoch)
            self.writer.add_scalar('Epoch_Loss/Wasserstein Distance', self.Wasserstein_D.avg, self.epoch)

            self.D_losses.reset()
            self.G_losses.reset()
            self.Wasserstein_D.reset()

            # print(self.config.dataset)
            if self.config.dataset != 'toy':
                # plt the generate images and loss curve
                self.save_results()

        if self.config.dataset != 'toy':
            create_gif(self.config.epoches, self.result_dir)


    def init_train(self):
        cudnn.benchmark = True
        self.start = time.time()

        traindir = self.args.train_dir


        if not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir)


        self.save_dir = os.path.join(self.args.save_dir, 'WGAN_'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.chkpt_dir = os.path.join(self.save_dir, 'chkpt')

        if not os.path.exists(self.chkpt_dir):
            os.makedirs(self.chkpt_dir)

        self.result_dir = os.path.join(self.save_dir, 'results')

        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        # setup loss function
        self.criterion = nn.BCELoss().cuda()

        # setup optimizer
        self.optimizerD = torch.optim.Adam(self.D.parameters(), lr=self.config.base_lr, betas=(self.config.beta1, 0.999))
        self.optimizerG = torch.optim.Adam(self.G.parameters(), lr=self.config.base_lr, betas=(self.config.beta1, 0.999))

        # setup some variables
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.D_losses = AverageMeter()
        self.G_losses = AverageMeter()
        self.Wasserstein_D = AverageMeter()

        fixed_noise = torch.FloatTensor(8 * 8, self.config.z_size, 1, 1).normal_(0, 1)
        self.fixed_noise = Variable(fixed_noise.cuda())


        self.one = torch.tensor(1.).to('cuda')
        # print(self.one.shape)
        self.mone = -1 * self.one

        self.D.train()
        self.G.train()
        self.iters = 0
        self.D_loss_list = []
        self.G_loss_list = []


    def save_results(self):
        plot_result(self.G, self.fixed_noise, self.config.image_size, self.epoch + 1, self.result_dir,
                    is_gray=(self.config.channel_size == 1))
        plot_loss(self.D_loss_list, self.G_loss_list, self.epoch + 1, self.config.epoches, self.result_dir)
        # save the D and G.
        save_checkpoint({'epoch': self.epoch, 'state_dict': self.D.state_dict(), },
                        os.path.join(self.chkpt_dir, 'D_epoch_{}'.format(self.epoch)))
        save_checkpoint({'epoch': self.epoch, 'state_dict': self.G.state_dict(), },
                        os.path.join(self.chkpt_dir, 'G_epoch_{}'.format(self.epoch)))


    def generate_grid(self, modelpath):
        self.G = load_checkpoint(self.G, modelpath)
        fig_size = (8, 8)
        image_size = self.config.image_size
        is_gray = self.config.channel_size == 1
        self.G.eval()
        with torch.no_grad():
            generate_images = self.G(torch.FloatTensor(8 * 8, self.config.z_size, 1, 1).normal_(0, 1))

        n_rows = n_cols = 8
        fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)

        for ax, img in zip(axes.flatten(), generate_images):
            ax.axis('off')
            # ax.set_adjustable('box-forced')
            if is_gray:
                img = img.cpu().data.view(image_size, image_size).numpy()
                ax.imshow(img, cmap='gray', aspect='equal')
            else:
                img = (((img - img.min()) * 255) / (img.max() - img.min())).cpu().data.numpy().transpose(1, 2,
                                                                                                         0).astype(
                    np.uint8)
                ax.imshow(img, cmap=None, aspect='equal')
        plt.subplots_adjust(wspace=0, hspace=0)
        title = 'Generated Samples'
        fig.text(0.5, 0.04, title, ha='center')

        plt.show()

    def generate(self, modelpath, n):
        self.G = load_checkpoint(self.G, modelpath)
        # fig_size = (8, 8)
        image_size = self.config.image_size
        is_gray = self.config.channel_size == 1
        self.G.eval()
        with torch.no_grad():
            noise = torch.FloatTensor(n, self.config.z_size, 1, 1).normal_(0, 1)
            generate_images = self.G(noise)

        for img in generate_images:
            if is_gray:
                img = img.cpu().data.view(image_size, image_size).numpy()
            else:
                img = (((img - img.min()) * 255) / (img.max() - img.min())).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)

        # print(type(generate_images))
        # print(generate_images.shape)
        return (noise, img)

    def interpolate(self, modelpath):
        self.G = load_checkpoint(self.G, modelpath)
        fig_size = (10, 2)
        image_size = self.config.image_size
        is_gray = self.config.channel_size == 1
        self.G.eval()
        with torch.no_grad():
            noise_seed = torch.FloatTensor(2, self.config.z_size, 1, 1).normal_(0, 1)
            noise = torch.zeros(11,self.config.z_size,1,1)
            for i in range(11):
                noise[i] = (i * noise_seed[0] + (10-i) * noise_seed[1]) / 10
            generate_images = self.G(noise)

        n_rows = n_cols = 8
        fig, axes = plt.subplots(1, 11, figsize=fig_size)

        for ax, img in zip(axes.flatten(), generate_images):
            ax.axis('off')
            # ax.set_adjustable('box-forced')
            if is_gray:
                img = img.cpu().data.view(image_size, image_size).numpy()
                ax.imshow(img, cmap='gray', aspect='equal')
            else:
                img = (((img - img.min()) * 255) / (img.max() - img.min())).cpu().data.numpy().transpose(1, 2,
                                                                                                         0).astype(
                    np.uint8)
                ax.imshow(img, cmap=None, aspect='equal')
        plt.subplots_adjust(wspace=0, hspace=0)
        title = 'Interpolation'
        fig.text(0.5, 0.04, title, ha='center')
        plt.show()

    def calc_gradient_penalty(self, real_data, fake_data, batch_size):
        # print(real_data.shape)
        # print(fake_data.shape)
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.unsqueeze(2).unsqueeze(3).cuda()
        # print(alpha.shape)
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = Variable(interpolates.cuda(), requires_grad=True)


        disc_interpolates = self.D(interpolates)
        # print(disc_interpolates.requires_grad)
        # print(interpolates.requires_grad)
        gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(self.config.batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.config.LAMBDA

        return gradient_penalty

    def data_dist(self, true_dist):
        N_POINTS = 128
        RANGE = 3
        true_dist = true_dist.data.cpu().numpy()
        points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
        points[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
        points[:, :, 1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
        points = torch.tensor(points.reshape((-1, 2))).cuda()

        disc_map = self.D(points).cpu().data.numpy()
        noise = torch.randn(self.config.batch_size, 2).cuda()
        samples = self.G(noise).data.cpu().numpy()

        plt.clf()

        x = y = np.linspace(-RANGE, RANGE, N_POINTS)
        plt.contour(x, y, disc_map.reshape((len(x), len(y))).transpose())

        plt.scatter(true_dist[:, 0], true_dist[:, 1], c='orange', marker='+')
        plt.scatter(samples[:, 0], samples[:, 1], c='green', marker='+')
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        image = PIL.Image.open(buf)
        # print(type(image))
        image = np.asarray(image)
        # print(image.shape)
        self.writer.add_image('Data Distribution', image, self.iters, dataformats='HWC')
        # print('Wrote Image to Tensorboard')


def construct_model(config):

    G = generator_noBN(z_size=config.z_size, out_size=config.channel_size, ngf=config.ngf).cuda()
    print('G network structure')
    print(G)
    D = discriminator_noBN(in_size=config.channel_size, ndf=config.ndf).cuda()
    print('D network structure')
    print(D)
    return (G, D)


