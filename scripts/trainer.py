import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.network import Generator, GlobalDis
from utils.logger import get_logger
from torch.autograd import Variable
from math import exp

logger = get_logger()


class Trainer(nn.Module):
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config
        self.use_cuda = self.config['cuda']
        self.device_ids = self.config['gpu_ids']

        self.netG = Generator(self.config['netG'], self.use_cuda, self.device_ids)
        self.globalD = GlobalDis(self.config['netD'], self.use_cuda, self.device_ids)

        self.optimizer_g = torch.optim.Adam(self.netG.parameters(), lr=self.config['lr'],
                                            betas=(self.config['beta1'], self.config['beta2']))
        d_params = list(self.globalD.parameters())
        self.optimizer_d = torch.optim.Adam(d_params, lr=config['lr'],
                                            betas=(self.config['beta1'], self.config['beta2']))

        if self.use_cuda:
            self.netG.to(self.device_ids[0])
            self.globalD.to(self.device_ids[0])

        self.ssim = SSIM()

    def forward(self, x, masks, ground_truth):
        self.train()
        losses = {}

        x1, x2 = self.netG(x, masks)
        x1_inpaint = x1 * masks + x * (1. - masks)
        x2_inpaint = x2 * masks + x * (1. - masks)

        ## D part
        refine_real, refine_fake = self.dis_forward(self.globalD, ground_truth, x2_inpaint.detach())
        losses['d_loss_loren'] = torch.mean(torch.log(1.0 + torch.abs(refine_real - refine_fake)))
        losses['d_loss_rel'] = (torch.mean(
            torch.nn.ReLU()(1.0 - (refine_real - torch.mean(refine_fake)))) + torch.mean(
            torch.nn.ReLU()(1.0 + (refine_fake - torch.mean(refine_real))))) / 2

        ## G part
        l1 = nn.L1Loss()(x1 * (1. - masks), ground_truth * (1. - masks)) * self.config['coarse_l1_alpha'] \
             + nn.L1Loss()(x2 * (1. - masks), ground_truth * (1. - masks))
        ssim = ((1. - self.ssim(ground_truth, x1_inpaint)) + (1.0 - self.ssim(ground_truth, x2_inpaint))) / 2.0
        losses['l1'] = l1 * 0.75 + ssim * 0.25

        refine_real, refine_fake = self.dis_forward(self.globalD, ground_truth, x2_inpaint)
        losses['g_loss_loren'] = torch.mean(torch.log(1.0 + torch.abs(refine_fake - refine_real)))
        losses['g_loss_rel'] = (torch.mean(
            torch.nn.ReLU()(1.0 + (refine_real - torch.mean(refine_fake)))) + torch.mean(
            torch.nn.ReLU()(1.0 - (refine_fake - torch.mean(refine_real))))) / 2

        return losses, x1_inpaint, x2_inpaint

    def dis_forward(self, netD, ground_truth, x_inpaint):
        assert ground_truth.size() == x_inpaint.size()
        batch_size = ground_truth.size(0)
        batch_data = torch.cat([ground_truth, x_inpaint], dim=0)
        batch_output = netD(batch_data)
        real_pred, fake_pred = torch.split(batch_output, batch_size, dim=0)

        return real_pred, fake_pred

    def save_model(self, checkpoint_dir):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(checkpoint_dir, 'gen.pt')
        global_dis_name = os.path.join(checkpoint_dir, 'global_dis.pt')
        gen_opt_name = os.path.join(checkpoint_dir, 'gen_optimizer.pt')
        torch.save(self.netG.state_dict(), gen_name)
        torch.save(self.globalD.state_dict(), global_dis_name)
        torch.save(self.optimizer_g.state_dict(), gen_opt_name)

    def resume(self, checkpoint_dir, iteration=1):
        g_checkpoint = torch.load(f'{checkpoint_dir}/gen.pt')
        global_dis_checkpoint = torch.load(f'{checkpoint_dir}/global_dis.pt')
        self.netG.load_state_dict(g_checkpoint, strict=False)
        self.globalD.load_state_dict(global_dis_checkpoint)
        print("Model loaded")
        return iteration



def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)