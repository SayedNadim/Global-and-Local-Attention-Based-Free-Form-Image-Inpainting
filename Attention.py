import torch
import torch.nn as nn
from utils.tools import *


class GlobalLocalAttention(nn.Module):
    def __init__(self, in_dim, ksize=3, stride=1, rate=1, fuse_k=3,down_rate = 8, softmax_scale=10,
                 fuse=True, use_cuda=True, device_ids=None):
        super(GlobalLocalAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.fuse_k = fuse_k
        self.softmax_scale = softmax_scale
        self.fuse = fuse
        self.chanel_in = in_dim
        self.down_rate = down_rate
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels= in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.query_conv_p = nn.Conv2d(in_channels=in_dim, out_channels= in_dim//8, kernel_size=1)
        self.key_conv_p = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv_p = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.beta = torch.nn.Parameter(torch.Tensor([1.0]))
        self.beta_2 = torch.nn.Parameter(torch.Tensor([1.0]))
        self.beta.requires_grad = True
        self.beta_2.requires_grad = True
    def forward(self, f, b, mask=None):
        # get shapes
        raw_int_fs = list(f.size())   # b*c*h*w
        raw_int_bs = list(b.size())   # b*c*h*w

        # extract patches from background with stride and rate
        kernel = 2 * self.rate
        # raw_w is extracted for reconstruction
        raw_w = extract_image_patches(b, ksizes=[kernel, kernel],
                                      strides=[self.rate*self.stride,
                                               self.rate*self.stride],
                                      rates=[1, 1],
                                      padding='same') # [N, C*k*k, L]
        # raw_shape: [N, C, k, k, L]
        raw_w = raw_w.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        raw_w = raw_w.permute(0, 4, 1, 2, 3)    # raw_shape: [N, L, C, k, k]
        raw_w_groups = torch.split(raw_w, 1, dim=0)

        # downscaling foreground option: downscaling both foreground and
        # background for matching and use original background for reconstruction.
        f = F.interpolate(f, scale_factor=1./self.rate, mode='nearest')
        b = F.interpolate(b, scale_factor=1./self.rate, mode='nearest')
        int_fs = list(f.size())     # b*c*h*w
        int_bs = list(b.size())
        f_groups = torch.split(f, 1, dim=0)  # split tensors along the batch dimension
        b_groups = torch.split(b, 1, dim=0) # split tensors along the batch dimension
        m_groups = torch.split(mask, 1, dim=0) # split tensors along the batch dimension
        mask = m_groups[0]
        # w shape: [N, C*k*k, L]
        w = extract_image_patches(b, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')
        # w shape: [N, C, k, k, L]
        w = w.view(int_bs[0], int_bs[1], self.ksize, self.ksize, -1)
        w = w.permute(0, 4, 1, 2, 3)    # w shape: [N, L, C, k, k]
        w_groups = torch.split(w, 1, dim=0)

        fw = extract_image_patches(f, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')
        # w shape: [N, C, k, k, L]
        fw = fw.view(int_fs[0], int_fs[1], self.ksize, self.ksize, -1)
        fw = fw.permute(0, 4, 1, 2, 3)    # w shape: [N, L, C, k, k]
        fw_groups = torch.split(fw, 1, dim=0)

        # process mask
        if mask is None:
            mask = torch.zeros([int_bs[0], 1, int_bs[2], int_bs[3]])
            if self.use_cuda:
                mask = mask.cuda()
        else:
            mask = F.interpolate(mask, scale_factor=1./(self.down_rate), mode='nearest')
        int_ms = list(mask.size())
        # m shape: [N, C*k*k, L]
        m = extract_image_patches(mask, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')
        # m shape: [N, C, k, k, L]
        m = m.view(int_ms[0], int_ms[1], self.ksize, self.ksize, -1)
        m = m.permute(0, 4, 1, 2, 3)    # m shape: [N, L, C, k, k]
        m = m[0]    # m shape: [L, C, k, k]
        # mm shape: [L, 1, 1, 1]
        mm = (reduce_mean(m, axis=[1, 2, 3], keepdim=True)==0.).to(torch.float32)
        mm = mm.permute(1, 0, 2, 3) # mm shape: [1, L, 1, 1]

        y = []
        offsets = []
        k = self.fuse_k
        scale = self.softmax_scale    # to fit the PyTorch tensor image value range
        fuse_weight = torch.eye(k).view(1, 1, k, k)  # 1*1*k*k
        fuse_weight = fuse_weight.cuda()

        for xi, bi, fi, wi, raw_wi in zip(f_groups, b_groups, fw_groups, w_groups, raw_w_groups):
            escape_NaN = torch.FloatTensor([1e-4])
            escape_NaN = escape_NaN.cuda()

            # Selecting patches
            fi = fi[0]
            wi = wi[0]
            #Patch Level Global Attention
            m_batchsize_p, C_p, width_p, height_p = fi.size()
            proj_query_p = self.query_conv_p(fi).view(m_batchsize_p, -1, width_p * height_p).permute(0, 2, 1)  # B, C, N -> B N C
            proj_key_p = self.key_conv(wi).view(m_batchsize_p, -1, width_p * height_p)  # B, C, N
            feature_similarity_p = torch.bmm(proj_query_p, proj_key_p)  # B, N, N

            mask_raw_p = m.view(m_batchsize_p, -1, width_p * height_p)  # B, C, N
            feature_pruning_p = feature_similarity_p * mask_raw_p
            attention_p = F.softmax(feature_pruning_p, dim=-1)  # B, N, C
            feature_final_p = torch.bmm(self.value_conv_p(fi).view(m_batchsize_p, -1, width_p * height_p),
                                      attention_p.permute(0, 2, 1))  # -. B, C, N
            final_pruning_p = feature_final_p.view(m_batchsize_p, C_p, width_p, height_p)  # B, C, H, W
            final_pruning_p = self.beta_2 * fi * m + (1.0 - m) * final_pruning_p

            max_wi = torch.max(torch.sqrt(reduce_sum(torch.pow(final_pruning_p, 2), axis=[1, 2, 3], keepdim=True)), escape_NaN)
            wi_normed = final_pruning_p / max_wi
            # max_xi = torch.max(torch.sqrt(reduce_sum(torch.pow(xi, 2), axis=[1, 2, 3], keepdim=True)), escape_NaN)
            # xi_normed = xi / max_xi
            # xi shape: [1, C, H, W], yi shape: [1, L, H, W]

            # # Global Attention
            # m_batchsize, C, width, height = xi_normed.size()  # B, C, H, W
            # proj_query = self.query_conv(xi_normed).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B, C, N -> B N C
            # proj_key = self.key_conv(bi).view(m_batchsize, -1, width * height)  # B, C, N
            # feature_similarity = torch.bmm(proj_query, proj_key)  # B, N, N
            #
            # mask_raw = mask.view(m_batchsize, -1, width * height)  # B, C, N
            # mask_raw = mask_raw.repeat(1, height * width, 1).permute(0, 2, 1)  # B, 1, H, W -> B, C, H, W //
            # feature_pruning = feature_similarity * mask_raw
            # attention = F.softmax(feature_pruning, dim = -1)  # B, N, C
            # feature_final = torch.bmm(self.value_conv(xi_normed).view(m_batchsize, -1, width * height),
            #                           attention.permute(0, 2, 1))  # -. B, C, N
            # final_pruning = feature_final.view(m_batchsize, C, width, height)  # B, C, H, W
            # final_pruning = self.beta * xi_normed * mask + (1.0 - mask) * final_pruning
            xi = same_padding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])  # xi: 1*c*H*W
            yi = F.conv2d(xi, wi_normed, stride=1)  # [1, L, H, W]
            if self.fuse:
                # make all of depth to spatial resolution
                yi = yi.view(1, 1, int_bs[2]*int_bs[3], int_fs[2]*int_fs[3])  # (B=1, I=1, H=32*32, W=32*32)
                yi = same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)  # (B=1, C=1, H=32*32, W=32*32)
                yi = yi.contiguous().view(1, int_bs[2], int_bs[3], int_fs[2], int_fs[3])  # (B=1, 32, 32, 32, 32)
                yi = yi.permute(0, 2, 1, 4, 3)
                yi = yi.contiguous().view(1, 1, int_bs[2]*int_bs[3], int_fs[2]*int_fs[3])
                yi = same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)
                yi = yi.contiguous().view(1, int_bs[3], int_bs[2], int_fs[3], int_fs[2])
                yi = yi.permute(0, 2, 1, 4, 3).contiguous()
            yi = yi.view(1, int_bs[2] * int_bs[3], int_fs[2], int_fs[3])  # (B=1, C=32*32, H=32, W=32)
            # # softmax to match
            yi = yi * mm
            yi = F.softmax(yi*scale, dim=1)
            yi = yi * mm
            # deconv for patch pasting
            wi_center = raw_wi[0]
            yi = F.conv_transpose2d(yi, wi_center, stride=self.rate, padding=1) / 4.  # (B=1, C=128, H=64, W=64)
            y.append(yi)

        y = torch.cat(y, dim=0)  # back to the mini-batch
        # y = F.pad(y, [0, 1, 0, 1])    # here may need conv_transpose same padding
        y.contiguous().view(raw_int_fs)

        return y

class GlobalAttention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(GlobalAttention, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)  #
        self.rate = 1
        self.gamma = torch.nn.Parameter(torch.Tensor([1.0]))
        self.gamma.requires_grad = True

    def forward(self, a, b, c):
        """
            inputs :
                x : input feature maps( B X C X W X H)
                c : B * 1 * W * H
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = a.size()  # B, C, H, W
        down_rate = int(c.size(2)//width)
        c = F.interpolate(c, scale_factor=1./down_rate*self.rate, mode='nearest')
        proj_query = self.query_conv(a).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B, C, N -> B N C
        proj_key = self.key_conv(b).view(m_batchsize, -1, width * height)  # B, C, N
        feature_similarity = torch.bmm(proj_query, proj_key)  # B, N, N

        mask = c.view(m_batchsize, -1, width * height)  # B, C, N
        mask = mask.repeat(1, height * width, 1).permute(0, 2, 1)  # B, 1, H, W -> B, C, H, W // B

        feature_pruning = feature_similarity * mask
        attention = self.softmax(feature_pruning)  # B, N, C

        # feature_similarity * mask
        feature_pruning = torch.bmm(self.value_conv(a).view(m_batchsize, -1, width * height),
                                    attention.permute(0, 2, 1))  # -. B, C, N
        out = feature_pruning.view(m_batchsize, C, width, height)  # B, C, H, W
        out = self.gamma * a*c + (1.0- c) * out
        return out