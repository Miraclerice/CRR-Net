'''
TransMatch

Original code retrieved from:
https://github.com/tzayuan/TransMatch_TMI

Original paper:
Chen Z, Zheng Y, Gee JC (2024) TransMatch: A transformer-based multilevel dual-stream feature matching network for unsupervised deformable image registration. IEEE Trans Med Imaging 431:15-27. https://doi.org/10.1109/TMI.2023.3288136

Modified and tested by:
Bingxian Xie
'''
import torch.nn as nn
import torch.nn.functional as F
import torch
from .Conv3dReLU import Conv3dReLU
from .LWSA import LWSA
from .LWCA import LWCA
from .Decoder import DecoderBlock, RegistrationHead
from .config import get_TransMatch_LPBA40_config

###########################################
# VoxelMorph basic network
###########################################
class SpatialTransformerBlock(nn.Module):

    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode

    def forward(self, src, flow):
        shape = flow.shape[2:]

        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        grid = grid.to(flow.device)

        new_locs = grid + flow
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
class TransMatch(nn.Module):
    def __init__(self, args=None, **kwargs):
        super(TransMatch, self).__init__()

        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        self.c1 = Conv3dReLU(2, 48, 3, 1, use_batchnorm=False)
        self.c2 = Conv3dReLU(2, 16, 3, 1, use_batchnorm=False)

        config2 = get_TransMatch_LPBA40_config()
        self.moving_lwsa = LWSA(config2)
        self.fixed_lwsa = LWSA(config2)

        self.lwca1 = LWCA(config2, dim_diy=96)
        self.lwca2 = LWCA(config2, dim_diy=192)
        self.lwca3 = LWCA(config2, dim_diy=384)
        self.lwca4 = LWCA(config2, dim_diy=768)

        self.up0 = DecoderBlock(768, 384, skip_channels=384, use_batchnorm=False)
        self.up1 = DecoderBlock(384, 192, skip_channels=192, use_batchnorm=False)
        self.up2 = DecoderBlock(192, 96, skip_channels=96, use_batchnorm=False)
        self.up3 = DecoderBlock(96, 48, skip_channels=48, use_batchnorm=False)
        self.up4 = DecoderBlock(48, 16, skip_channels=16, use_batchnorm=False)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.reg_head = RegistrationHead(
            in_channels=48,
            out_channels=3,
            kernel_size=3,
        )
        self.stn = SpatialTransformerBlock(mode='bilinear')

    def forward(self, moving_Input, fixed_Input):

        input_fusion = torch.cat((moving_Input, fixed_Input), dim=1)

        x_s1 = self.avg_pool(input_fusion)

        f4 = self.c1(x_s1)
        f5 = self.c2(input_fusion)

        B, _, _, _, _ = moving_Input.shape  # Batch, channel, height, width, depth

        moving_fea_4, moving_fea_8, moving_fea_16, moving_fea_32 = self.moving_lwsa(moving_Input)
        fixed_fea_4, fixed_fea_8, fixed_fea_16, fixed_fea_32 = self.moving_lwsa(fixed_Input)

        moving_fea_4_cross = self.lwca1(moving_fea_4, fixed_fea_4)
        moving_fea_8_cross = self.lwca2(moving_fea_8, fixed_fea_8)
        moving_fea_16_cross = self.lwca3(moving_fea_16, fixed_fea_16)
        moving_fea_32_cross = self.lwca4(moving_fea_32, fixed_fea_32)

        fixed_fea_4_cross = self.lwca1(fixed_fea_4, moving_fea_4)
        fixed_fea_8_cross = self.lwca2(fixed_fea_8, moving_fea_8)
        fixed_fea_16_cross = self.lwca3(fixed_fea_16, moving_fea_16)
        # fixed_fea_32_cross = self.lwca4(fixed_fea_32, moving_fea_32)


        x = self.up0(moving_fea_32_cross, moving_fea_16_cross, fixed_fea_16_cross)
        x = self.up1(x, moving_fea_8_cross, fixed_fea_8_cross)
        x = self.up2(x, moving_fea_4_cross, fixed_fea_4_cross)
        x = self.up3(x, f4)
        x = self.up(x)
        flow = self.reg_head(x)
        warped = self.stn(moving_Input, flow)

        return warped, flow

if __name__ == '__main__':
    m = TransMatch(args=None)
    data = torch.randn(1, 1, 160, 192, 160)
    output = m(data, data)
    print(output.shape)
    params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(params / 1e6)
