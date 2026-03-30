# -*- coding: utf-8 -*-
# @Author: MiracleRice
# Blog   : miraclerice.top
import os

import numpy as np
from medpy import metric
import nibabel as nib
import natsort

def dice_val(y_pred, y_true, C=None, exclude_background=True):
    if C is None:
        C = np.unique(y_true)
    if exclude_background:
        C = [i for i in C if i != 0]
    total = []
    for i in C:
        if np.any(y_pred == i) and np.any(y_true == i):
            total.append(metric.dc(y_pred == i, y_true == i))
        else:
            print(f'Label {i} not in prediction or ground truth')
    total.append(np.mean(total))
    return total

def assd_val(y_pred, y_true, C=None, exclude_background=True):
    if C is None:
        C = np.unique(y_true)
    if exclude_background:
        C = [i for i in C if i != 0]
    total = []
    for i in C:
        if np.any(y_pred == i) and np.any(y_true == i):
            total.append(metric.assd(y_pred == i, y_true == i, np.ones(3)))
        else:
            print(f'Label {i} not in prediction or ground truth')
    total.append(np.mean(total))
    return total



def hd95_val(y_pred, y_true, C=None, exclude_background=True):
    if C is None:
        C = np.unique(y_true)
    if exclude_background:
        C = [i for i in C if i != 0]
    total = []
    for i in C:
        if np.any(y_pred == i) and np.any(y_true == i):
            total.append(metric.hd95(y_pred == i, y_true == i))
        else:
            print(f'Label {i} not in prediction or ground truth')
    total.append(np.mean(total))
    return total

"""
Batch Processing Statistics Report:
Number of successfully processed pairs: 380 / 380 pair
Average time per pair: 695.21 seconds
Variance in time taken per pair: 80016.19 (seconds^2)
Standard deviation in time per pair: 282.87 seconds
Total elapsed time: 4403.01 minutes
695.21 ± 282.87 s
"""

if __name__ == '__main__':
    path = r'path/to/Mindboggle/'
    pairs = natsort.natsorted(os.listdir(path))
    raw_dices = []
    deformed_dices = []
    raw_hd95s = []
    deformed_hd95s = []
    raw_assds = []
    deformed_assds = []
    for i, pair in enumerate(pairs):
        print(f'pair{i + 1}'.center(100, '-'))
        src_seg = nib.load(fr'{path}\{pair}\x_seg.nii').get_fdata().astype('int')
        tar_seg = nib.load(fr'{path}\{pair}\y_seg.nii').get_fdata().astype('int')
        wsrc_seg = nib.load(fr'{path}\{pair}\w_x_seg.nii').get_fdata().astype('int')
        raw_dsc = dice_val(src_seg, tar_seg)[-1]
        print('raw dice:\t', raw_dsc, end='\t')
        deformed_dsc = dice_val(wsrc_seg, tar_seg)[-1]
        print('deformed dice:\t', deformed_dsc)
        raw_dices.append(raw_dsc)
        deformed_dices.append(deformed_dsc)

        raw_hd95 = hd95_val(src_seg, tar_seg)[-1]
        print('raw hd95:\t', raw_hd95, end='\t')
        deformed_hd95 = hd95_val(wsrc_seg, tar_seg)[-1]
        print('deformed hd95:\t', deformed_hd95)
        raw_hd95s.append(raw_hd95)
        deformed_hd95s.append(deformed_hd95)

        raw_assd = assd_val(src_seg, tar_seg)[-1]
        print('raw assd:\t', raw_assd, end='\t')
        deformed_assd = assd_val(wsrc_seg, tar_seg)[-1]
        print('deformed assd:\t', deformed_assd)
        raw_assds.append(raw_assd)
        deformed_assds.append(deformed_assd)

    print('Final Statistics'.center(100, '-'))
    print('raw dice:\t', np.mean(raw_dices), f'({np.std(raw_dices)})', end='\t')
    print('deformed dice:\t', np.mean(deformed_dices), f'({np.std(deformed_dices)})')
    print('raw hd95:\t', np.mean(raw_hd95s), f'({np.std(raw_hd95s)})', end='\t')
    print('deformed hd95:\t', np.mean(deformed_hd95s), f'({np.std(deformed_hd95s)})')
    print('raw assd:\t', np.mean(raw_assds), f'({np.std(raw_assds)})', end='\t')
    print('deformed assd:\t', np.mean(deformed_assds), f'({np.std(deformed_assds)})')

"""
------------------------------------------Final Statistics------------------------------------------
raw dice:	 0.31746305431158495 (0.023770741119103062)	deformed dice:	 0.4924265812182998 (0.015158821471694867)
raw hd95:	 7.124282163039762 (0.6116309519399231)	deformed hd95:	 5.810594426131797 (0.42967558735827016)
raw assd:	 2.3131241004934564 (0.2461576395613167)	deformed assd:	 1.5832469157932634 (0.10941993772366994)
"""
