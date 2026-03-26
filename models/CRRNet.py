# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import typing as t
from torch.distributions.normal import Normal
import math
from timm.models.layers import DropPath, trunc_normal_tf_
from timm.models import named_apply
from functools import partial


"""
SpatialTransformer, ResizeTransformerBlock, and VecIntBlock implement warping, upsamping, and integrating operations respectively.
Original code sourced from: https://github.com/voxelmorph/voxelmorph/blob/dev/voxelmorph/nn/modules.py
Modified and tested by: Bingxian Xie
Orignal paper: 
@article{balakrishnan2019voxelmorph,
  title={Voxelmorph: a learning framework for deformable medical image registration},
  author={Balakrishnan, Guha and Zhao, Amy and Sabuncu, Mert R and Guttag, John and Dalca, Adrian V},
  journal={IEEE transactions on medical imaging},
  volume={38},
  number={8},
  pages={1788--1800},
  year={2019},
  publisher={IEEE}
}
"""
class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode
        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class ResizeTransformerBlock(nn.Module):
    def __init__(self, resize_factor, mode='trilinear'):
        super().__init__()
        self.factor = resize_factor
        self.mode = mode

    def forward(self, x):
        if self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        elif self.factor < 1:
            # resize first to save memory
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x
        return x


class VecIntBlock(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, size, nsteps=7):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(size, mode='bilinear')

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec

def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def voxel_shuffle(self: torch.Tensor, upscale_factor: int):
    """
    A 3D version for voxel rearrangement of correlation features
    Original code sourced from: https://github.com/pytorch/pytorch/blob/90618581e971d28ac6950305d72521af05ed3a42/torch/_refs/nn/functional/__init__.py#L1230-L1253
    Modified and tested by: Bingxian Xie
    Orignal paper: 
    @inproceedings{shi2016real,
        title={Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network},
        author={Shi, Wenzhe and Caballero, Jose and Husz{\'a}r, Ferenc and Totz, Johannes and Aitken, Andrew P and Bishop, Rob and Rueckert, Daniel and Wang, Zehan},
        booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
        pages={1874--1883},
        year={2016}
    }  
    """
    assert self.dim() >= 4, f"voxel_shuffle expects input to have at least 4 dimensions, but got input with {self.dim} dimension(s)"
    batch = self.shape[:-4]
    C_out = self.shape[-4] // upscale_factor ** 3
    HWD_out = (self.shape[-3] * upscale_factor, self.shape[-2] * upscale_factor, self.shape[-1] * upscale_factor)
    n = len(batch)
    B_dims = range(n)
    C_dim, r1_dim, r2_dim, r3_dim, H_dim, W_dim, D_dim = range(n, n + 7)
    self = self.contiguous()
    # print(self.is_contiguous())
    return (
        self.view(
            *batch,
            C_out,
            upscale_factor,
            upscale_factor,
            upscale_factor,
            self.shape[-3],
            self.shape[-2],
            self.shape[-1], )
        .permute(*B_dims, C_dim, H_dim, r1_dim, W_dim, r2_dim, D_dim, r3_dim)
        .reshape(*batch, C_out, *HWD_out))


def voxel_unshuffle(self: torch.Tensor, downscale_factor: int):
    """
    A 3D version for reverse voxel rearrangement of correlation features
    """
    assert self.dim() >= 4, f"voxel_shuffle expects input to have at least 4 dimensions, but got input with {self.dim} dimension(s)"
    batch = self.shape[:-4]
    C_out = self.shape[-4] * downscale_factor ** 3
    HWD_out = (
    self.shape[-3] // downscale_factor, self.shape[-2] // downscale_factor, self.shape[-1] // downscale_factor)
    n = len(batch)
    B_dims = range(n)
    C_dim, H_dim, r1_dim, W_dim, r2_dim, D_dim, r3_dim = range(n, n + 7)
    self = self.contiguous()
    # print(self.is_contiguous())
    return (
        self.view(
            *batch,
            self.shape[-4],
            HWD_out[0],
            downscale_factor,
            HWD_out[1],
            downscale_factor,
            HWD_out[2],
            downscale_factor,
        )
        .permute(*B_dims, C_dim, r1_dim, r2_dim, r3_dim, H_dim, W_dim, D_dim)
        .reshape(*batch, C_out, *HWD_out))


class ConvInsBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 norm_layer=nn.InstanceNorm3d, act_layer=nn.LeakyReLU, alpha=0.2, bias=True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.norm = norm_layer(out_channels)
        self.act = act_layer(alpha)
        self.conv.weight = nn.Parameter(Normal(0, 1e-2).sample(self.conv.weight.shape))
        self.conv.bias = nn.Parameter(torch.zeros(self.conv.bias.shape))

    def forward(self, x_in):
        x_out = self.conv(x_in)
        x_out = self.norm(x_out)
        x_out = self.act(x_out)
        return x_out


class RegHeadBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.reg_head = nn.Conv3d(in_channels, 3, kernel_size=3, stride=1, padding=1)
        self.reg_head.weight = nn.Parameter(Normal(0, 1e-5).sample(self.reg_head.weight.shape))
        self.reg_head.bias = nn.Parameter(torch.zeros(self.reg_head.bias.shape))

    def forward(self, x_in):
        x_out = self.reg_head(x_in)
        return x_out


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, alpha=0.1):
        super(UpConvBlock, self).__init__()

        self.upconv = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(alpha, inplace=True),
        )

    def forward(self, x_in):
        x_out = self.upconv(x_in)
        return x_out

class Correlation3D(nn.Module):
    def __init__(self, kernel_size=3, cuda_acc=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.cuda_acc = cuda_acc

    def forward(self, x_1, x_2):
        B, C, H, W, D = x_1.size()
        pd = self.kernel_size - 1
        pr = pd // 2
        x_2 = F.pad(x_2, pad=(pr, pr, pr, pr, pr, pr), value=0)
        if self.cuda_acc:
            from CorrCuda import correlation
            x_2.reshape(B, C, H + pd, W + pd, D + pd).permute(0, 2, 3, 4, 1)
            x_1 = x_1.permute(0, 2, 3, 4, 1) / C
            x_out = correlation.correlation_cu(x_1, x_2, self.kernel_size)
            x_out = x_out.permute(0, 4, 1, 2, 3)
            print(x_out.shape)
            print(x_out.is_contiguous())
        else:
            # pytorch nnf.unfold 5D is not currently supported
            # x_2 = x_2.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1).unfold(4, self.kernel_size, 1).permute(
            #     0, 5, 6, 7, 1, 2, 3, 4) # more slices
            # x_2 = x_2.reshape(B, -1, C, H, W, D)
            # x_out = torch.mean(x_1.unsqueeze(1) * x_2, 2) # or sum
            offsetx, offsety, offsetz = torch.meshgrid([torch.arange(0, self.kernel_size),
                                                        torch.arange(0, self.kernel_size),
                                                        torch.arange(0, self.kernel_size)], indexing='ij')
            x_out = torch.cat([torch.mean(x_1 * x_2[:, :, dx:dx + H, dy:dy + W, dz:dz + D], 1, keepdim=True)
                               for dx, dy, dz in zip(offsetx.reshape(-1), offsety.reshape(-1), offsetz.reshape(-1))], 1)
        return x_out

class ChannelAttention(nn.Module):
    """
    Adapted the 2D channel attention mechanism to a 3D version for refining correaltion representtations
    Original code sourced from: https://github.com/Lose-Code/UBRFC-Net/blob/master/UBRFC/Attention.py
    Modified and tested by: Bingxian Xie
    Orignal paper: 
    @article{sun2024unsupervised,
        title={Unsupervised bidirectional contrastive reconstruction and adaptive fine-grained channel attention networks for image dehazing},
        author={Sun, Hang and Wen, Yang and Feng, Huijing and Zheng, Yuelin and Mei, Qi and Ren, Dong and Yu, Mei},
        journal={Neural Networks},
        volume={176},
        pages={106314},
        year={2024},
        publisher={Elsevier}
    }
    """
    def __init__(self, channel, b=1, gamma=2, w=0.5):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.fc = nn.Conv3d(channel, channel, 1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.w = nn.Parameter(torch.FloatTensor([w]), requires_grad=True)

    def forward(self, input):
        x = self.avg_pool(input)
        x1 = self.conv1(x.squeeze(-1).squeeze(-1).transpose(-1, -2)).transpose(-1, -2)
        x2 = self.fc(x).squeeze(-1).squeeze(-1).transpose(-1, -2)  # (b,1,c)
        out1 = torch.sum(torch.matmul(x1, x2), dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        out1 = self.sigmoid(out1)
        out2 = torch.sum(torch.matmul(x2.transpose(-1, -2), x1.transpose(-1, -2)), dim=1).unsqueeze(-1).unsqueeze(
            -1).unsqueeze(-1)

        out2 = self.sigmoid(out2)
        mix_factor = self.sigmoid(self.w)
        out = out1 * mix_factor.expand_as(out1) + out2 * (1 - mix_factor.expand_as(out2))
        out = self.conv1(out.squeeze(-1).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)
        out = self.sigmoid(out)
        return input * out

class SpatialAttention(nn.Module):
    """
    Adapted the 2D spatial attention mechanism to a 3D version for refining correaltion representtations
    Original code sourced from: https://github.com/HZAI-ZJNU/SCSA/blob/main/mmpretrain/models/attentions/SCSA.py
    Modified and tested by: Bingxian Xie
    Orignal paper: 
    @article{si2025scsa,
        title={SCSA: Exploring the synergistic effects between spatial and channel attention},
        author={Si, Yunzhong and Xu, Huiying and Zhu, Xinzhong and Zhang, Wenhao and Dong, Yao and Chen, Yuxing and Li, Hongbo},
        journal={Neurocomputing},
        pages={129866},
        year={2025},
        publisher={Elsevier}
    }
    """
    def __init__(self, input_channels, group_kernel_sizes=[1, 3, 5, 7]):
        super().__init__()
        self.dim = input_channels

        self.group_chans = group_chans = self.dim // 4
        self.local_dwc = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[0],
                                   padding=group_kernel_sizes[0] // 2, groups=group_chans)
        self.global_dwc_s = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[1],
                                      padding=group_kernel_sizes[1] // 2, groups=group_chans)
        self.global_dwc_m = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[2],
                                      padding=group_kernel_sizes[2] // 2, groups=group_chans)
        self.global_dwc_l = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[3],
                                      padding=group_kernel_sizes[3] // 2, groups=group_chans)
        self.norm_h = nn.GroupNorm(4, self.dim)
        self.norm_w = nn.GroupNorm(4, self.dim)
        self.norm_d = nn.GroupNorm(4, self.dim)
        self.act = nn.Sigmoid()

    def forward(self, x_in):
        # The dim of x is (B, C, H, W, D)
        # Channel attention
        b, c, h_, w_, d_ = x_in.size()
        # Spatial attention
        # (B, C, H)
        x_h = x_in.mean(dim=[3, 4])
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans, dim=1)
        # (B, C, W)
        x_w = x_in.mean(dim=[2, 4])
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans, dim=1)
        # (B, C, D)
        x_d = x_in.mean(dim=[2, 3])
        l_x_d, g_x_d_s, g_x_d_m, g_x_d_l = torch.split(x_d, self.group_chans, dim=1)

        x_h_attn = self.act(self.norm_h(torch.cat((
            self.local_dwc(l_x_h),
            self.global_dwc_s(g_x_h_s),
            self.global_dwc_m(g_x_h_m),
            self.global_dwc_l(g_x_h_l),
        ), dim=1)))
        x_h_attn = x_h_attn.view(b, c, h_, 1, 1)

        x_w_attn = self.act(self.norm_w(torch.cat((
            self.local_dwc(l_x_w),
            self.global_dwc_s(g_x_w_s),
            self.global_dwc_m(g_x_w_m),
            self.global_dwc_l(g_x_w_l)
        ), dim=1)))
        x_w_attn = x_w_attn.view(b, c, 1, w_, 1)

        x_d_attn = self.act(self.norm_d(torch.cat((
            self.local_dwc(l_x_d),
            self.global_dwc_s(g_x_d_s),
            self.global_dwc_m(g_x_d_m),
            self.global_dwc_l(g_x_d_l)
        ), dim=1)))
        x_d_attn = x_d_attn.view(b, c, 1, 1, d_)

        x_sa = x_in * x_h_attn * x_w_attn * x_d_attn
        return x_sa


class CSBlock(nn.Module):
    def __init__(self, input_channels, group_kernel_sizes=[1, 3, 5, 7]):
        super().__init__()
        self.dim = input_channels
        self.ca = ChannelAttention(input_channels) # FCA
        self.sa = SpatialAttention(input_channels=self.dim, group_kernel_sizes=group_kernel_sizes)

    def forward(self, x_in):
        # The dim of x is (B, C, H, W, D)
        x_sc = x_in
        x_ca = self.ca(x_in)
        x_out = x_ca + x_sc
        x_out = self.sa(x_out) 
        x_out = x_out + x_sc
        return x_out

class FRM(nn.Module):
    # Feature Reconstruction
    def __init__(self, in_feats, out_feats, scale_factor=2, bias=True):
        super().__init__()
        self.scale_factor = scale_factor
        self.shuffle_dim = out_feats // scale_factor ** 3
        self.conv1 = ConvInsBlock(in_feats, out_feats, kernel_size=3, padding=1, bias=bias)
        self.sConv = ConvInsBlock(self.shuffle_dim, self.shuffle_dim, kernel_size=3, padding=1, bias=bias)
        self.usConv = ConvInsBlock(out_feats, out_feats, kernel_size=3, padding=1, bias=bias)

    def forward(self, x_in):
        x_sc = self.conv1(x_in)  # shortcut
        x_out = voxel_shuffle(x_sc, self.scale_factor)
        x_out = self.sConv(x_out)
        x_out = voxel_unshuffle(x_out, self.scale_factor)
        x_out = self.usConv(x_out) + x_sc
        return x_out


class CRRM(nn.Module):
    def __init__(self, in_c, out_c, calc_corr=True, corr_k=3, cuda_acc=False, use_frm=True, use_attn=True, group_kernel_sizes=[1, 3, 5, 7]):
        super().__init__()
        self.calc_corr = calc_corr
        self.m_c = in_c * 2
        self.use_attn = use_attn
        if self.calc_corr:
            self.corr_channels = corr_k ** 3
            self.corr_layer = Correlation3D(corr_k, cuda_acc=cuda_acc)
            self.m_c = in_c * 2 + self.corr_channels
        if use_frm:
            self.frm = FRM(self.m_c, out_c)
        else:
            self.frm = nn.Conv3d(self.m_c, out_c, kernel_size=3, padding=1)
        if use_attn:
            self.cs_blocks = CSBlock(out_c, group_kernel_sizes)

    def forward(self, x_1, x_2):
        if self.calc_corr:
            x_corr = self.corr_layer(x_1, x_2)
            x_out = self.frm(torch.cat([x_1, x_corr, x_2], dim=1))
        else:
            x_out = self.frm(torch.cat([x_1, x_2], dim=1))
        if self.use_attn:
            x_out = self.cs_blocks(x_out)
        return x_out

class Encoder(nn.Module):
    def __init__(self, in_channels, feats):
        super().__init__()
        self.conv1 = nn.Sequential(
            ConvInsBlock(in_channels, feats),
            ConvInsBlock(feats, feats)
        )
        self.conv2 = nn.Sequential(
            nn.AvgPool3d(2, 2),
            ConvInsBlock(feats, 2 * feats, stride=1),
            ConvInsBlock(2 * feats, 2 * feats, stride=1)
        )
        self.conv3 = nn.Sequential(
            nn.AvgPool3d(2, 2),
            ConvInsBlock(2 * feats, 4 * feats, stride=1),
            ConvInsBlock(4 * feats, 4 * feats, stride=1)
        )
        self.conv4 = nn.Sequential(
            nn.AvgPool3d(2, 2),
            ConvInsBlock(4 * feats, 8 * feats, stride=1),
            ConvInsBlock(8 * feats, 8 * feats, stride=1)
        )

    def forward(self, x_in):
        x1 = self.conv1(x_in)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        return x1, x2, x3, x4


class CRR_Net(nn.Module):
    def __init__(self, in_shape=[160, 192, 160], in_channels=1, feats=8, calc_corr=True, corr_k=3, use_frm=True, use_attn=True, cuda_acc=True, group_kernel_sizes=[1, 3, 5, 7],  **kwargs):
        super().__init__()
        self.encoder = Encoder(in_channels, feats)
        # step1
        self.dec_stage11 = CRRM(feats, feats * 2, calc_corr=calc_corr, cuda_acc=cuda_acc, corr_k=corr_k, use_frm=use_frm, use_attn=use_attn, group_kernel_sizes=group_kernel_sizes)
        self.dec_stage12 = CRRM(feats * 2, feats * 4, calc_corr=calc_corr, cuda_acc=cuda_acc, corr_k=corr_k, use_frm=use_frm, use_attn=use_attn, group_kernel_sizes=group_kernel_sizes)
        self.dec_stage13 = CRRM(feats * 4, feats * 8, calc_corr=calc_corr, cuda_acc=cuda_acc, corr_k=corr_k, use_frm=use_frm, use_attn=use_attn, group_kernel_sizes=group_kernel_sizes)
        self.dec_stage14 = CRRM(feats * 8, feats * 16, calc_corr=calc_corr, cuda_acc=cuda_acc, corr_k=corr_k, use_frm=use_frm, use_attn=use_attn, group_kernel_sizes=group_kernel_sizes)
        # step2
        self.dec_stage21 = CRRM(feats * 2, feats * 2, calc_corr=calc_corr, cuda_acc=cuda_acc, corr_k=corr_k, use_frm=use_frm, use_attn=use_attn, group_kernel_sizes=group_kernel_sizes)
        self.dec_stage22 = CRRM(feats * 4, feats * 4, calc_corr=calc_corr, cuda_acc=cuda_acc, corr_k=corr_k, use_frm=use_frm, use_attn=use_attn, group_kernel_sizes=group_kernel_sizes)
        self.dec_stage23 = CRRM(feats * 8, feats * 8, calc_corr=calc_corr, cuda_acc=cuda_acc, corr_k=corr_k, use_frm=use_frm, use_attn=use_attn, group_kernel_sizes=group_kernel_sizes)

        # deformable registration head
        self.reghead1 = RegHeadBlock(feats * 2)
        self.reghead2 = RegHeadBlock(feats * 4)
        self.reghead3 = RegHeadBlock(feats * 8)
        self.reghead4 = RegHeadBlock(feats * 16)

        self.upsample1 = UpConvBlock(feats * 4, feats * 2)
        self.upsample2 = UpConvBlock(feats * 8, feats * 4)
        self.upsample3 = UpConvBlock(feats * 16, feats * 8)

        self.resizeTransformer = ResizeTransformerBlock(resize_factor=2, mode='trilinear')
        self.stn = nn.ModuleList([
            SpatialTransformer([s // 2 ** i for s in in_shape])
            for i in range(3)
        ])

    def forward(self, m, f):
        x_m = self.encoder(m)
        x_f = self.encoder(f)
        # decoder
        x_m1, x_m2, x_m3, x_m4 = x_m
        x_f1, x_f2, x_f3, x_f4 = x_f
        # stage1
        x4 = self.dec_stage14(x_m4, x_f4)
        varphi = self.reghead4(x4)
        flow = self.resizeTransformer(varphi)

        # stage2
        x_m3 = self.stn[2](x_m3, flow)
        x = self.dec_stage13(x_m3, x_f3)
        x3 = self.dec_stage23(x, self.upsample3(x4))
        varphi = self.reghead3(x3)
        flow = self.resizeTransformer(self.stn[2](flow, varphi) + varphi)

        # stage3
        x_m2 = self.stn[1](x_m2, flow)
        x = self.dec_stage12(x_m2, x_f2)
        x2 = self.dec_stage22(x, self.upsample2(x3))
        varphi = self.reghead2(x2)
        flow = self.resizeTransformer(self.stn[1](flow, varphi) + varphi)

        # stage4
        x_m1 = self.stn[0](x_m1, flow)
        x = self.dec_stage11(x_m1, x_f1)
        x1 = self.dec_stage21(x, self.upsample1(x2))
        varphi = self.reghead1(x1)
        flow = self.stn[0](flow, varphi) + varphi

        warped = self.stn[0](m, flow)
        return warped, flow

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    in_shape = [160, 192, 160]
    m = CRR_Net(in_shape, cuda_acc=False).to(device)
    x1 = torch.randn(1, 1, *in_shape).to(device)
    x2 = torch.randn(1, 1, *in_shape).to(device)
    out = m(x1, x2)
    for i in out:
        print(i.shape)
    from thop import profile
    macs, params = profile(m, inputs=(torch.randn(1, 1, *in_shape, device=device), torch.randn(1, 1, *in_shape,device=device)))
    print(macs / 1e9, params / 1e6)
