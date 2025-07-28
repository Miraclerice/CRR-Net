# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pystrum.pynd.ndutils as nd
from medpy import metric
import SimpleITK as sitk

def get_LPBAlabel():
    label_names = [
        'L superior frontal gyrus',
        'R superior frontal gyrus',
        'L middle frontal gyrus',
        'R middle frontal gyrus',
        'L inferior frontal gyrus',
        'R inferior frontal gyrus',
        'L precentral gyrus',
        'R precentral gyrus',
        'L middle orbitofrontal gyrus',
        'R middle orbitofrontal gyrus',
        'L lateral orbitofrontal gyrus',
        'R lateral orbitofrontal gyrus',
        'L gyrus rectus',
        'R gyrus rectus',
        'L postcentral gyrus',
        'R postcentral gyrus',
        'L superior parietal gyrus',
        'R superior parietal gyrus',
        'L supramarginal gyrus',
        'R supramarginal gyrus',
        'L angular gyrus',
        'R angular gyrus',
        'L precuneus',
        'R precuneus',
        'L superior occipital gyrus',
        'R superior occipital gyrus',
        'L middle occipital gyrus',
        'R middle occipital gyrus',
        'L inferior occipital gyrus',
        'R inferior occipital gyrus',
        'L cuneus',
        'R cuneus',
        'L superior temporal gyrus',
        'R superior temporal gyrus',
        'L middle temporal gyrus',
        'R middle temporal gyrus',
        'L inferior temporal gyrus',
        'R inferior temporal gyrus',
        'L parahippocampal gyrus',
        'R parahippocampal gyrus',
        'L lingual gyrus',
        'R lingual gyrus',
        'L fusiform gyrus',
        'R fusiform gyrus',
        'L insular cortex',
        'R insular cortex',
        'L cingulate gyrus',
        'R cingulate gyrus',
        'L caudate',
        'R caudate',
        'L putamen',
        'R putamen',
        'L hippocampus',
        'R hippocampus'
    ]
    return label_names, len(label_names)

def get_SimpleLPBAlabel():
    target_labels = ['Frontal', 'Parietal', 'Occipital', 'Temporal', 'Fusiform', 'Putamen', 'Hippocampus']
    return target_labels, len(target_labels)

def getSimpleLabelIndices():
    labels, _ = get_LPBAlabel()
    target_labels, _ = get_SimpleLPBAlabel()
    res_indices = []
    for key in target_labels:
        key_indices = [i for i, label in enumerate(labels) if key.lower() in label.lower()]
        res_indices.append(key_indices)
    return res_indices

def save_np2nii_with_referance(np_img, save_path, ref_itk):
    nii_img = sitk.GetImageFromArray(np_img)
    # ref_nii = sitk.ReadImage(ref_path)
    nii_img.CopyInformation(ref_itk)
    sitk.WriteImage(nii_img, save_path)


def dice_val(y_pred, y_true, C=None):
    C = C or np.unique(y_true).shape[0]
    total = []
    for i in range(1, C):
        if np.any(y_pred == i) and np.any(y_true == i):
            total.append(metric.dc(y_pred == i, y_true == i))
        else:
            print(f'Label {i} not in prediction or ground truth')
    total.append(np.mean(total))
    return total


def assd_val(y_pred, y_true, C=None):
    C = C or np.unique(y_true).shape[0]
    total = []
    for i in range(1, C):
        if np.any(y_pred == i) and np.any(y_true == i):
            total.append(metric.assd(y_pred == i, y_true == i, np.ones(3)))
        else:
            print(f'Label {i} not in prediction or ground truth')
    total.append(np.mean(total))
    return total


def hd_val(y_pred, y_true, C=None):
    C = C or np.unique(y_true).shape[0]
    total = []
    for i in range(1, C):
        if np.any(y_pred == i) and np.any(y_true == i):
            total.append(metric.hd(y_pred == i, y_true == i))
        else:
            print(f'Label {i} not in prediction or ground truth')
    total.append(np.mean(total))
    return total


def hd95_val(y_pred, y_true, C=None):
    C = C or np.unique(y_true).shape[0]
    total = []
    for i in range(1, C):
        if np.any(y_pred == i) and np.any(y_true == i):
            total.append(metric.hd95(y_pred == i, y_true == i))
        else:
            print(f'Label {i} not in prediction or ground truth')
    total.append(np.mean(total))
    return total

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)


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
        grid = grid.type(torch.FloatTensor).cuda()

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

class register_model(nn.Module):
    def __init__(self, img_size=(64, 256, 256), mode='bilinear'):
        super(register_model, self).__init__()
        self.spatial_trans = SpatialTransformer(img_size, mode)

    def forward(self, x):
        img = x[0].cuda()
        flow = x[1].cuda()
        out = self.spatial_trans(img, flow)
        return out


def dice_val_VOI(y_pred, y_true):
    VOI_lbls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                48, 49, 50, 51, 52, 53, 54]

    pred = y_pred.detach().cpu().numpy()[0, 0, ...]
    true = y_true.detach().cpu().numpy()[0, 0, ...]
    DSCs = np.zeros((len(VOI_lbls), 1))
    idx = 0
    for i in VOI_lbls:
        pred_i = pred == i
        true_i = true == i
        intersection = pred_i * true_i
        intersection = np.sum(intersection)
        union = np.sum(pred_i) + np.sum(true_i)
        dsc = (2.*intersection) / (union + 1e-5)
        DSCs[idx] =dsc
        idx += 1
    return np.mean(DSCs)

def jacobian_determinant_vxm(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    disp = disp.transpose(1, 2, 3, 0)
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]


def bw_grid(vol_shape, spacing, thickness=1):
    """
    draw a black and white ND grid.

    Parameters
    ----------
        vol_shape: expected volume size(slice size)
        spacing: scalar or list the same size as vol_shape

    Returns
    -------
        grid_vol: a volume the size of vol_shape with white lines on black background
    """

    # check inputs
    if not isinstance(spacing, (list, tuple)):
        spacing = [spacing] * len(vol_shape)
    spacing = [f + 1 for f in spacing]
    assert len(vol_shape) == len(spacing)

    # go through axes
    grid_image = np.zeros(vol_shape)
    for d, v in enumerate(vol_shape):
        rng = [np.arange(0, f) for f in vol_shape]
        for t in range(thickness):
            rng[d] = np.append(np.arange(0 + t, v, spacing[d]), -1)
            grid_image[tuple(ndgrid(*rng))] = 1

    return grid_image


def ndgrid(*args, **kwargs):
    """
    Disclaimer: This code is taken directly from the scitools package [1]
    Since at the time of writing scitools predominantly requires python 2.7 while we work with 3.5+
    To avoid issues, we copy the quick code here.

    Same as calling ``meshgrid`` with *indexing* = ``'ij'`` (see
    ``meshgrid`` for documentation).
    """
    kwargs['indexing'] = 'ij'
    return np.meshgrid(*args, **kwargs)


def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    """Creating regular gird, no batch size and channel"""
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[0], grid_step):
        grid_img[j + line_thickness - 1, :, :] = 1
    for i in range(0, grid_img.shape[1], grid_step):
        grid_img[:, i + line_thickness - 1, :] = 1
    return grid_img
