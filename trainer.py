import os

import torch
import torch.nn as nn
from torch import autograd

from model.networks_chord import Generator, GlobalDis
from utils.logger import get_logger
from utils.tools import get_model_list, local_patch, spatial_discounting_mask
from ssim import SSIM

logger = get_logger()


class Trainer(nn.Module):
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config
        self.use_cuda = self.config['cuda']
        self.device_ids = self.config['gpu_ids']

        self.netG = Generator(self.config['netG'], self.use_cuda, self.device_ids)
        # self.localD = LocalDis(self.config['netD'], self.use_cuda, self.device_ids)
        self.globalD = GlobalDis(self.config['netD'], self.use_cuda, self.device_ids)

        self.optimizer_g = torch.optim.Adam(self.netG.parameters(), lr=self.config['lr'],
                                            betas=(self.config['beta1'], self.config['beta2']))
        d_params = list(self.globalD.parameters())
        self.optimizer_d = torch.optim.Adam(d_params, lr=config['lr'],
                                            betas=(self.config['beta1'], self.config['beta2']))

        if self.use_cuda:
            self.netG.to(self.device_ids[0])
            # self.localD.to(self.device_ids[0])
            self.globalD.to(self.device_ids[0])

        self.ssim = SSIM()

    def forward(self, x, bboxes, masks, ground_truth):
        self.train()
        losses = {}

        x1, x2 = self.netG(x, masks)
        # local_patch_gt = local_patch(ground_truth, bboxes)
        x1_inpaint = x1 * masks + x * (1. - masks)
        x2_inpaint = x2 * masks + x * (1. - masks)
        # local_patch_x1_inpaint = local_patch(x1_inpaint, bboxes)
        # local_patch_x2_inpaint = local_patch(x2_inpaint, bboxes)

        ## D part

        # local_patch_real_pred, local_patch_fake_pred = \
        #     self.dis_forward(self.localD, local_patch_gt, local_patch_x2_inpaint.detach())
        # global_real_pred, global_fake_pred = \
        #     self.dis_forward(self.globalD, ground_truth, x2_inpaint.detach())
        # losses['wgan_d'] = torch.mean(local_patch_fake_pred - local_patch_real_pred) \
        #                    + torch.mean(global_fake_pred - global_real_pred) * self.config['global_wgan_loss_alpha']
        # # gradients penalty loss
        # local_penalty = self.calc_gradient_penalty(self.localD, local_patch_gt, local_patch_x2_inpaint.detach())
        # global_penalty = self.calc_gradient_penalty(self.globalD, ground_truth, x2_inpaint.detach())
        # losses['wgan_gp'] = local_penalty + global_penalty
        # coarse_real, coarse_fake = self.dis_forward(self.globalD, ground_truth, x1_inpaint.detach())
        refine_real, refine_fake = self.dis_forward(self.globalD, ground_truth, x2_inpaint.detach())
        losses['d_loss_loren'] = torch.mean(torch.log(1.0 + torch.abs(refine_real- refine_fake)))
        # losses['d_loss_rel_local'] = (torch.mean(
        #     torch.nn.ReLU()(1.0 - (local_patch_real_pred - torch.mean(local_patch_fake_pred)))) + torch.mean(
        #     torch.nn.ReLU()(1.0 + (local_patch_fake_pred - torch.mean(local_patch_real_pred))))) / 2
        # losses['d_loss_rel_global'] = (torch.mean(
        #     torch.nn.ReLU()(1.0 - (global_real_pred - torch.mean(global_fake_pred)))) + torch.mean(
        #     torch.nn.ReLU()(1.0 + (global_fake_pred - torch.mean(global_real_pred))))) / 2
        losses['d_loss_rel'] = (torch.mean(
            torch.nn.ReLU()(1.0 - (refine_real - torch.mean(refine_fake)))) + torch.mean(
            torch.nn.ReLU()(1.0 + (refine_fake - torch.mean(refine_real))))) / 2
        ## G part
        # sd_mask = spatial_discounting_mask(self.config)
        # losses['l1'] = nn.L1Loss()(local_patch_x1_inpaint * sd_mask, local_patch_gt * sd_mask) \
        #                * self.config['coarse_l1_alpha'] \
        #                + nn.L1Loss()(local_patch_x2_inpaint * sd_mask, local_patch_gt * sd_mask)
        l1 = nn.L1Loss()(x1 * (1. - masks), ground_truth * (1. - masks)) * self.config['coarse_l1_alpha'] \
                       + nn.L1Loss()(x2 * (1. - masks), ground_truth * (1. - masks))
        ssim = ((1. - self.ssim(ground_truth, x1_inpaint)) + ( 1.0 - self.ssim(ground_truth,  x2_inpaint)))/2.0
        losses['l1'] = l1 * 0.75 + ssim * 0.25

        # # wgan g loss
        # local_patch_real_pred, local_patch_fake_pred = \
        #     self.dis_forward(self.localD, local_patch_gt, local_patch_x2_inpaint)
        # global_real_pred, global_fake_pred = self.dis_forward(self.globalD, ground_truth, x2_inpaint)
        #
        # losses['wgan_g'] = -torch.mean(local_patch_fake_pred) \
        #                    - torch.mean(global_fake_pred) * self.config['global_wgan_loss_alpha']
        # losses['g_loss_rel_local'] = (torch.mean(
        #     torch.nn.ReLU()(1.0 + (local_patch_real_pred - torch.mean(local_patch_fake_pred)))) + torch.mean(
        #     torch.nn.ReLU()(1.0 - (local_patch_fake_pred - torch.mean(local_patch_real_pred))))) / 2
        # losses['g_loss_rel_global'] = (torch.mean(
        #     torch.nn.ReLU()(1.0 + (global_real_pred - torch.mean(global_fake_pred)))) + torch.mean(
        #     torch.nn.ReLU()(1.0 - (global_fake_pred - torch.mean(global_real_pred))))) / 2
        # losses['g_loss_rel'] = (torch.mean(
        #     torch.nn.ReLU()(1.0 + (ground_truth - torch.mean(x2_inpaint)))) + torch.mean(
        #     torch.nn.ReLU()(1.0 - (x2_inpaint - torch.mean(ground_truth))))) / 2 + (torch.mean(
        #     torch.nn.ReLU()(1.0 + (ground_truth - torch.mean(x1_inpaint)))) + torch.mean(
        #     torch.nn.ReLU()(1.0 - (x1_inpaint - torch.mean(ground_truth))))) / 2
        # losses['g_loss_loren'] = torch.mean(torch.log(1.0 + torch.abs(x2_inpaint - ground_truth)))
        # losses['g_loss_global'] = (torch.mean(
        #     torch.nn.ReLU()(1.0 + (global_real_pred - torch.mean(global_fake_pred)))) + torch.mean(
        #     torch.nn.ReLU()(1.0 - (global_fake_pred - torch.mean(global_real_pred))))) / 2
        # losses['g_loss_loren'] = torch.mean(torch.log(1.0 + torch.abs(local_patch_fake_pred-local_patch_real_pred)))
        # losses['g_loss_local'] = (torch.mean(
        #     torch.nn.ReLU()(1.0 + (local_patch_real_pred - torch.mean(local_patch_fake_pred)))) + torch.mean(
        #     torch.nn.ReLU()(1.0 - (local_patch_fake_pred - torch.mean(local_patch_real_pred))))) / 2


        # coarse_real, coarse_fake = self.dis_forward(self.globalD, ground_truth, x1_inpaint)
        refine_real, refine_fake = self.dis_forward(self.globalD, ground_truth, x2_inpaint)
        losses['g_loss_loren'] = torch.mean(torch.log(1.0 + torch.abs(refine_fake- refine_real)))
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

    # Calculate gradient penalty
    def calc_gradient_penalty(self, netD, real_data, fake_data):
        batch_size, channel, height, width = real_data.size()
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand(batch_size, int(real_data.nelement() // batch_size)).contiguous() \
            .view(batch_size, channel, height, width)
        if self.use_cuda:
            alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates)

        grad_outputs = torch.ones(disc_interpolates.size())
        if self.use_cuda:
            grad_outputs = grad_outputs.cuda()
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=grad_outputs,
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    def inference(self, x, masks):
        self.eval()
        x1, x2, offset_flow = self.netG(x, masks)
        # x1_inpaint = x1 * masks + x * (1. - masks)
        x2_inpaint = x2 * masks + x * (1. - masks)

        return x2_inpaint, offset_flow

    def save_model(self, checkpoint_dir):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(checkpoint_dir, 'gen.pt')
        # local_dis_name = os.path.join(checkpoint_dir, 'local_dis.pt')
        global_dis_name = os.path.join(checkpoint_dir, 'global_dis.pt')
        gen_opt_name = os.path.join(checkpoint_dir, 'gen_optimizer.pt')
        # dis_opt_name = os.path.join(checkpoint_dir, 'dis_optimizer.pt')
        torch.save(self.netG.state_dict(), gen_name)
        # torch.save(self.localD.state_dict(), local_dis_name)
        torch.save(self.globalD.state_dict(), global_dis_name)
        torch.save(self.optimizer_g.state_dict(), gen_opt_name)
        # torch.save(self.optimizer_d.state_dict(), dis_opt_name)

    # def resume(self, checkpoint_dir, iteration=0, test=False):
    #     # Load generators
    #     last_model_name = get_model_list(checkpoint_dir, "gen", iteration=iteration)
    #     self.netG.load_state_dict(torch.load(last_model_name))
    #     # iteration = int(last_model_name[-11:-3])
    #
    #     if not test:
    #         # Load discriminators
    #         last_model_name = get_model_list(checkpoint_dir, "dis")  # , iteration=iteration
    #         state_dict = torch.load(last_model_name)
    #         self.localD.load_state_dict(state_dict['localD'])
    #         self.globalD.load_state_dict(state_dict['globalD'])
    #         # Load optimizers
    #         state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
    #         self.optimizer_d.load_state_dict(state_dict['dis'])
    #         self.optimizer_g.load_state_dict(state_dict['gen'])
    #
    #     print("Resume from {}".format(checkpoint_dir))  # at iteration {} , iteration
    #     logger.info("Resume from {}".format(checkpoint_dir))  # at iteration {} , iteration
    #
    #     return iteration

    def resume(self, checkpoint_dir, iteration =1):
        g_checkpoint = torch.load(f'{checkpoint_dir}/gen.pt')
        # local_dis_checkpoint = torch.load(f'{checkpoint_dir}/local_dis.pt')
        global_dis_checkpoint = torch.load(f'{checkpoint_dir}/global_dis.pt')
        self.netG.load_state_dict(g_checkpoint)
        # self.localD.load_state_dict(local_dis_checkpoint)
        self.globalD.load_state_dict(global_dis_checkpoint)
        print("Model loaded")
        return iteration