# -*- coding: utf-8 -*-
import glob
import time
import os, utils
import random
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from natsort import natsorted
import yaml
import SimpleITK as sitk
from argparse import ArgumentParser
import importlib


def dynamic_import(module_name, name):
    module = importlib.import_module(module_name)
    return getattr(module, name)


def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True


same_seeds(24)


def main(cfg):
    name = cfg['name']
    val_dir = cfg['val_dir']
    model_idx = -1
    weights = cfg['weights']
    model_folder = f'{name}_ncc_{weights[0]}_reg_{weights[1]}/'
    model_dir = 'experiments/' + model_folder

    img_size = cfg['img_size']
    model = dynamic_import(cfg['model_path'], cfg['name'])(**cfg['model_args'])
    if 'ckpt' in cfg and cfg['ckpt']:
        ckpt = torch.load(cfg['ckpt'])
        print('Best model: {}'.format(cfg['ckpt']))
        model.load_state_dict(ckpt['state_dict'])
    else:
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
        print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
        model.load_state_dict(best_model)
    model.cuda()
    model.eval()
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()
    test_composed = transforms.Compose([trans.Seg_norm(),
                                        trans.NumpyType((np.float32, np.int16)),
                                        ])
    test_set = datasets.LPBAInferDataset(glob.glob(val_dir + '*.pkl'), transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    eval_dsc_def = utils.AverageMeter()
    eval_assd_def = utils.AverageMeter()
    eval_hd95_def = utils.AverageMeter()
    eval_det = utils.AverageMeter()
    infer_times = []
    warmup_model(model, test_loader, warmup_steps=10)
    with torch.no_grad():
        print('Infering...')
        stdy_idx = 0
        for data in test_loader:
            print("-" * 50)
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]
            torch.cuda.synchronize()
            start = time.time()
            x_def, flow = model(x, y)[:2]
            def_out = reg_model([x_seg.cuda().float(), flow.cuda()])
            torch.cuda.synchronize()
            end = time.time()
            infer_times.append(end - start)
            print(f'[{stdy_idx + 1}]Infer time:{end - start}s')
            tar = y.detach().cpu().numpy()[0, 0, :, :, :]

            jac_det = utils.jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :, :])
            dsc_trans = utils.dice_val_VOI(def_out.long(), y_seg.long())
            eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))
            eval_dsc_def.update(dsc_trans.item(), x.size(0))
            print('Trans dsc: {:.4f}'.format(dsc_trans.item()))
            print('Det < 0: {}'.format(np.sum(jac_det <= 0) / np.prod(tar.shape)))

            x_seg_np = x_seg.detach().cpu().numpy().squeeze()
            def_seg_np = def_out.detach().cpu().numpy().squeeze()
            y_seg_np = y_seg.detach().cpu().numpy().squeeze()

            # assd
            assd_trans = utils.assd_val(def_seg_np, y_seg_np)[-1]
            eval_assd_def.update(assd_trans.item(), x.size(0))
            print('Trans assd: {:.4f}'.format(assd_trans.item()))

            # hd95
            hd95_trans = utils.hd95_val(def_seg_np, y_seg_np)[-1]
            eval_hd95_def.update(hd95_trans.item(), x.size(0))
            print('Trans hd95: {:.4f}'.format(hd95_trans.item()))
            stdy_idx += 1

        print(" LPBA40 Results".center(50, '-'))
        print('Def dsc: {:.4f}, std: {:.4f}'.format(eval_dsc_def.avg, eval_dsc_def.std))
        print('Deformed det < 0: {}, std: {}'.format(eval_det.avg, eval_det.std))
        print('Def assd: {:.4f}, std: {:.4f}'.format(eval_assd_def.avg, eval_assd_def.std))
        print('Def hd95: {:.4f}, std: {:.4f}'.format(eval_hd95_def.avg, eval_hd95_def.std))
        print('Time: {:.4f}'.format(np.mean(infer_times)))

def warmup_model(model, data_loader, warmup_steps=10):
    print(f'Warm up the model for {warmup_steps} steps...')
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(data_loader):
            if idx >= warmup_steps:
                break
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            model(x, y)
    print('Warmup completed!')


if __name__ == '__main__':
    '''
    GPU configuration
    '''
    parser = ArgumentParser()
    parser.add_argument('-mp', '--model_path', type=str, default='models', help='path to network module')
    parser.add_argument('-m', '--model', type=str, default='CRR_Net', help='select model')
    args = parser.parse_args()
    with open('cfg.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)[args.model]
        cfg['model_path'] = args.model_path
        cfg['name'] = args.model
    print('Configurations are as follows: \n' + str(cfg))
    GPU_iden = cfg['gpu_id']
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main(cfg)
