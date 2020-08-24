import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.tools import *

# Contextual attention implementation is borrowed from IJCAI 2019 : "MUSICAL: Multi-Scale Image Contextual Attention Learning for Inpainting".
# Original implementation causes bad results for Pytorch 1.2+.
class GlobalLocalAttention(nn.Module):
    def __init__(self, in_dim, patch_size=3, propagate_size=3, stride=1):
        super(GlobalLocalAttention, self).__init__()
        self.patch_size = patch_size
        self.propagate_size = propagate_size
        self.stride = stride
        self.prop_kernels = None
        self.in_dim = in_dim
        self.feature_attention = GlobalAttention(in_dim)
        self.patch_attention = GlobalAttentionPatch(in_dim)

    def forward(self, foreground, mask, background="same"):
        ###assume the masked area has value 1
        bz, nc, w, h = foreground.size()
        if background == "same":
            background = foreground.clone()
        mask = F.interpolate(mask, size=(h, w), mode='nearest')
        background = background * (1 - mask)
        foreground = self.feature_attention(foreground, background, mask)
        background = F.pad(background,
                           [self.patch_size // 2, self.patch_size // 2, self.patch_size // 2, self.patch_size // 2])
        conv_kernels_all = background.unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size,
                                                                                     self.stride).contiguous().view(bz,
                                                                                                                    nc,
                                                                                                                    -1,
                                                                                                                    self.patch_size,
                                                                                                                    self.patch_size)

        mask_resized = mask.repeat(1, self.in_dim, 1, 1)
        mask_resized = F.pad(mask_resized,
                             [self.patch_size // 2, self.patch_size // 2, self.patch_size // 2, self.patch_size // 2])
        mask_kernels_all = mask_resized.unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size,
                                                                                       self.stride).contiguous().view(
            bz,
            nc,
            -1,
            self.patch_size,
            self.patch_size)
        conv_kernels_all = conv_kernels_all.transpose(2, 1)
        mask_kernels_all = mask_kernels_all.transpose(2, 1)
        output_tensor = []
        for i in range(bz):
            feature_map = foreground[i:i + 1]

            # form convolutional kernels
            conv_kernels = conv_kernels_all[i] + 0.0000001
            mask_kernels = mask_kernels_all[i]
            conv_kernels = self.patch_attention(conv_kernels, conv_kernels, mask_kernels)
            norm_factor = torch.sum(conv_kernels ** 2, [1, 2, 3], keepdim=True) ** 0.5
            conv_kernels = conv_kernels / norm_factor

            conv_result = F.conv2d(feature_map, conv_kernels, padding=self.patch_size // 2)
#             print(conv_result.shape)
            if self.propagate_size != 1:
                if self.prop_kernels is None:
                    self.prop_kernels = torch.ones([conv_result.size(1), 1, self.propagate_size, self.propagate_size])
                    self.prop_kernels.requires_grad = False
                    self.prop_kernels = self.prop_kernels.cuda()
                conv_result = F.conv2d(conv_result, self.prop_kernels, stride=1, padding=1, groups=conv_result.size(1))
            mm = (torch.mean(mask_kernels_all[i], dim=[1,2,3], keepdim=True)==0.0).to(torch.float32)
            mm = mm.permute(1,0,2,3).cuda()
            conv_result = conv_result * mm
            attention_scores = F.softmax(conv_result, dim=1)
            attention_scores = attention_scores * mm

            ##propagate the scores
            recovered_foreground = F.conv_transpose2d(attention_scores, conv_kernels, stride=1,
                                                      padding=self.patch_size // 2)
            output_tensor.append(recovered_foreground)
        return torch.cat(output_tensor, dim=0)


class GlobalAttention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(GlobalAttention, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)  #
        self.rate = 1
        self.gamma = torch.tensor([1.0], requires_grad=True).cuda()

    def forward(self, a, b, c):
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
        feature_pruning = torch.bmm(self.value_conv(a).view(m_batchsize, -1, width * height),
                                    attention.permute(0, 2, 1))  # -. B, C, N
        out = feature_pruning.view(m_batchsize, C, width, height)  # B, C, H, W
        out = a * c + self.gamma *  (1.0 - c) * out
        return out


class GlobalAttentionPatch(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(GlobalAttentionPatch, self).__init__()
        self.chanel_in = in_dim

        self.query_channel = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_channel = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_channel = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.softmax_channel = nn.Softmax(dim=-1)
        self.gamma = torch.tensor([1.0], requires_grad=True).cuda()

    def forward(self, x, y, m):
        '''
        Something
        '''
        feature_size = list(x.size())
        # Channel attention
        query_channel = self.query_channel(x).view(feature_size[0], -1, feature_size[2] * feature_size[3])
        key_channel = self.key_channel(y).view(feature_size[0], -1, feature_size[2] * feature_size[3]).permute(0,
                                                                                                               2,
                                                                                                               1)
        channel_correlation = torch.bmm(query_channel, key_channel)
        m_r = m.view(feature_size[0], -1, feature_size[2]*feature_size[3])
        channel_correlation = torch.bmm(channel_correlation, m_r)
        energy_channel = self.softmax_channel(channel_correlation)
        value_channel = self.value_channel(x).view(feature_size[0], -1, feature_size[2] * feature_size[3])
        attented_channel = (energy_channel * value_channel).view(feature_size[0], feature_size[1],
                                                                         feature_size[2],
                                                                         feature_size[3])
        out = x * m + self.gamma * (1.0 - m) * attented_channel
        return out


if __name__ == '__main__':
    x = torch.rand(3, 128, 64, 64, requires_grad=True).float().cuda()
    y = torch.rand(3, 1, 256, 256, requires_grad=False).float().cuda()
    y[y > 0.5] = 1
    y[y <= 0.5] = 0
    net = GlobalLocalAttention(128).cuda()
    out = net(x, y)
    print(out.shape)
