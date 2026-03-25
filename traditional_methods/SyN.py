import glob
import os, utils, torch
import sys, ants
sys.path.append('..')
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
from torchvision import transforms
import nibabel as nib
import SimpleITK as sitk
from utils import dice_val, assd_val, hd_val, hd95_val
import time

def nib_load(file_name):
    if not os.path.exists(file_name):
        return np.array([1])

    proxy = nib.load(file_name)
    data = proxy.get_fdata()
    proxy.uncache()  # free memory
    return data


def main():
    test_dir = '/media/zgm/c3c40470-006a-4c53-aa4a-9f50621c7edb/zgm/students/xbx/dataset/LPBA_data/Val/'
    test_composed = transforms.Compose([trans.Seg_norm(),
                                        trans.NumpyType((np.float32, np.int16)),
                                        ])
    test_set = datasets.LPBAInferDataset(glob.glob(test_dir + '*.pkl'), transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    eval_dsc_def = utils.AverageMeter()
    eval_dsc_raw = utils.AverageMeter()
    eval_assd_def = utils.AverageMeter()
    eval_assd_raw = utils.AverageMeter()
    eval_hd95_def = utils.AverageMeter()
    eval_hd95_raw = utils.AverageMeter()

    eval_det = utils.AverageMeter()
    with torch.no_grad():
        print('Infering...')
        print("ants.registration(y, x, 'SyNOnly', reg_iterations=(160, 80, 40), syn_metric='meansquares')")
        stdy_idx = 0
        for data in test_loader:
            print("-"*50)
            x = data[0].squeeze(0).squeeze(0).cpu().numpy()  # hwd
            y = data[1].squeeze(0).squeeze(0).cpu().numpy()  # hwd
            x_seg = data[2].squeeze(0).squeeze(0).cpu().numpy()  # hwd
            y_seg = data[3].squeeze(0).squeeze(0).cpu().numpy()  # hwd

            x = ants.from_numpy(x)
            y = ants.from_numpy(y)
            x_ants = ants.from_numpy(x_seg.astype(np.float32))
            y_ants = ants.from_numpy(y_seg.astype(np.float32))
            start = time.time()
            # https://antspyx.readthedocs.io/en/latest/registration.html
            reg_m2f = ants.registration(y, x, 'SyNOnly', reg_iterations=(160, 80, 40), syn_metric='meansquares')
            def_seg = ants.apply_transforms(fixed=y_ants, moving=x_ants, transformlist=reg_m2f['fwdtransforms'],
                                            interpolator='nearestNeighbor')
            print(f'[{stdy_idx + 1}]Infer time:{time.time() - start}s')
            def_src = reg_m2f['warpedmovout']
            flow = np.array(nib_load(reg_m2f['fwdtransforms'][0]), dtype='float32', order='C')
            flow = flow[:, :, :, 0, :].transpose(3, 0, 1, 2)
            x_seg = torch.from_numpy(x_seg[None, None, ...])
            def_seg = torch.from_numpy(def_seg.numpy()[None, None, ...])
            y_seg = torch.from_numpy(y_seg[None, None, ...])
            dsc_trans = utils.dice_val_VOI(def_seg.long(), y_seg.long())
            dsc_raw = utils.dice_val_VOI(x_seg.long(), y_seg.long())
            jac_det = utils.jacobian_determinant_vxm(flow)
            eval_dsc_def.update(dsc_trans.item(), 1)
            eval_dsc_raw.update(dsc_raw.item(), 1)
            eval_det.update(np.sum(jac_det <= 0) / np.prod(y_seg.shape), 1)
            print('DSC: {:.4f}, Raw dsc: {:.4f}'.format(dsc_trans.item(), dsc_raw.item()))
            print('Det < 0: {}'.format(np.sum(jac_det <= 0) / np.prod(y_seg.shape)))

            x_seg_np = x_seg.squeeze().long().numpy()
            def_seg_np = def_seg.squeeze().long().numpy()
            y_seg_np = y_seg.squeeze().long().numpy()
            # assd
            assd_trans = assd_val(def_seg_np, y_seg_np)[-1]
            assd_raw = assd_val(x_seg_np, y_seg_np)[-1]
            eval_assd_def.update(assd_trans, 1)
            eval_assd_raw.update(assd_raw, 1)
            # hd95
            hd95_trans = hd95_val(def_seg_np, y_seg_np)[-1]
            hd95_raw = hd95_val(x_seg_np, y_seg_np)[-1]
            eval_hd95_def.update(hd95_trans, 1)
            eval_hd95_raw.update(hd95_raw, 1)
            print('ASSD: {:.4f}, Raw assd: {:.4f}'.format(assd_trans, assd_raw))
            print('HD95: {:.4f}, Raw hd95: {:.4f}'.format(hd95_trans, hd95_raw))
            stdy_idx += 1
        print('Def dsc: {:.4f}, std: {:.4f}'.format(eval_dsc_def.avg, eval_dsc_def.std))
        print('Raw dsc: {:.4f}, std: {:.4f}'.format(eval_dsc_raw.avg, eval_dsc_raw.std))
        print('Det < 0 {}, std: {}'.format(eval_det.avg, eval_det.std))
        print('Def assd: {:.4f}, std: {:.4f}'.format(eval_assd_def.avg, eval_assd_def.std))
        print('Raw assd: {:.4f}, std: {:.4f}'.format(eval_assd_raw.avg, eval_assd_raw.std))
        print('Def hd95: {:.4f}, std: {:.4f}'.format(eval_hd95_def.avg, eval_hd95_def.std))
        print('Raw hd95: {:.4f}, std: {:.4f}'.format(eval_hd95_raw.avg, eval_hd95_raw.std))


def save_flow(src, tar, src_def, flow, path='./img_results'):
    src = src[0, 0, :, :, :]
    src_def = src_def[0, 0, :, :, :]
    flow = flow.transpose(1, 2, 3, 0)
    sitk.WriteImage(sitk.GetImageFromArray(src), path + '/src.nii.gz')
    sitk.WriteImage(sitk.GetImageFromArray(tar), path + '/tar.nii.gz')
    sitk.WriteImage(sitk.GetImageFromArray(src_def), path + '/src_def.nii.gz')
    sitk.WriteImage(sitk.GetImageFromArray(flow), path + '/flow.nii.gz')


if __name__ == '__main__':
    main()
