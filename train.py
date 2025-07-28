# -*- coding: utf-8 -*-

import os, utils, glob, losses
import sys
import random
import yaml
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import matplotlib.pyplot as plt
from natsort import natsorted
from argparse import ArgumentParser
import importlib
import warnings

warnings.filterwarnings("ignore")


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


class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir + "logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        # self.flush()

    def flush(self):
        pass
        # self.terminal.flush()
        # self.log.flush()


def main(cfg):
    name = cfg['name']
    batch_size = cfg['batch_size']
    train_dir = cfg['train_dir']
    val_dir = cfg['val_dir']
    weights = cfg['weights'] 
    save_dir = f'{name}_ncc_{weights[0]}_reg_{weights[1]}/'
    if 'ckpt' in cfg and cfg['ckpt']:
        save_dir = cfg['ckpt'].split('/')[1]+'/'
    if not os.path.exists('experiments/' + save_dir):
        os.makedirs('experiments/' + save_dir)
    if not os.path.exists('logs/' + save_dir):
        os.makedirs('logs/' + save_dir)
    if 'log' in cfg and cfg['log']: 
        sys.stdout = Logger('logs/' + save_dir) 
    f = open(os.path.join('logs/' + save_dir, 'losses_and_dice.txt'), 'a')

    lr = cfg['lr']
    epoch_start = cfg['epoch_start']
    max_epoch = cfg['max_epoch']
    cont_training = cfg['cont_training']
    img_size = cfg['img_size']

    '''
    Initialize model
    '''
    model = dynamic_import(cfg['model_path'], cfg['name'])(**cfg['model_args'])
    model.cuda()

    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()

    '''
    If continue from previous training
    '''
    if cont_training:
        ckpt = torch.load(cfg['ckpt'])
        epoch_start = ckpt['epoch']
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch, 0.9), 8)
        print('Model: {} loaded!'.format(cfg['ckpt']))
        model.load_state_dict(ckpt['state_dict'])
    else:
        updated_lr = lr

    '''
    Initialize training
    '''
    train_composed = transforms.Compose([trans.NumpyType((np.float32, np.float32))])

    val_composed = transforms.Compose([trans.Seg_norm(), 
                                       trans.NumpyType((np.float32, np.int16))])
    train_set = datasets.LPBADataset(glob.glob(train_dir + '*.pkl'), transforms=train_composed)
    val_set = datasets.LPBAInferDataset(glob.glob(val_dir + '*.pkl'), transforms=val_composed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    criterion = losses.NCC_vxm()
    criterions = [criterion]
    criterions += [losses.Grad3d(penalty='l2')]
    best_dsc = 0
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        idx = 0
        for data in train_loader: 
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            output = model(x, y)
            loss = 0
            loss_vals = []
            for n, loss_function in enumerate(criterions):
                curr_loss = loss_function(output[n], y) * weights[n]
                loss_vals.append(curr_loss)
                loss += curr_loss
            loss_all.update(loss.item(), y.numel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(idx, len(train_loader), loss.item(),
                                                                                   loss_vals[0].item(),
                                                                                   loss_vals[1].item()))

        print('[Epoch {}] loss: {:.4f}'.format(epoch, loss_all.avg))
        print('[Epoch {}] loss_avg: {:.4f} loss_std: {:.4f}'.format(epoch, loss_all.avg, loss_all.std), file=f, end=' ')
        '''
        Validation
        '''
        eval_dsc = utils.AverageMeter()
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                data = [t.cuda() for t in data]
                x = data[0]
                y = data[1]
                x_seg = data[2]
                y_seg = data[3]
                output = model(x, y)
                def_out = reg_model([x_seg.cuda().float(), output[1].cuda()])
                dsc = utils.dice_val_VOI(def_out.long(), y_seg.long())
                eval_dsc.update(dsc.item(), x.size(0))
                print(f'[Epoch {epoch}] DSC: {eval_dsc.avg:.4f}')
        best_dsc = max(eval_dsc.avg, best_dsc)
        print(f'DSC_avg: {eval_dsc.avg}, DSC_std: {eval_dsc.std}', file=f)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_dsc': best_dsc,
            'optimizer': optimizer.state_dict(),
        }, save_dir='experiments/' + save_dir, filename='dsc{:.3f}.pth.tar'.format(eval_dsc.avg))
        loss_all.reset()
    f.close()


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    """ Learning rate decay """
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, save_dir + filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))


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
    print(cfg)
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
