# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset
import numpy as np

def pkl_loader(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

class LPBADataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def __getitem__(self, index):
        # index: [0, 30*29]
        x_idx = index // (len(self.paths) - 1)
        s = index % (len(self.paths) - 1)
        y_idx = s + 1 if s >= x_idx else s
        path_x = self.paths[x_idx]
        path_y = self.paths[y_idx]
        x, x_seg = pkl_loader(path_x)  # fixed image
        y, y_seg = pkl_loader(path_y)  # moving image
        x, y = x[None, ...], y[None, ...]
        x, y = self.transforms([x, y])
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.paths) * (len(self.paths) - 1)


class LPBAInferDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def __getitem__(self, index):
        x_idx = index // (len(self.paths) - 1)
        s = index % (len(self.paths) - 1)
        y_idx = s + 1 if s >= x_idx else s
        path_x = self.paths[x_idx]
        path_y = self.paths[y_idx]
        x, x_seg = pkl_loader(path_x)  # moving image
        y, y_seg = pkl_loader(path_y)  # fixed image
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths) * (len(self.paths) - 1)

if __name__ == '__main__':
    import trans
    from torchvision import transforms
    import glob
    from torch.utils.data import DataLoader
    test_composed = transforms.Compose([trans.Simple_Seg_norm(),
                                        trans.NumpyType((np.float32, np.int16)),])
    test_set = LPBAInferDataset(glob.glob('/home/Anonymous author/dataset/LPBA40/Val/' + '*.pkl'), transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, pin_memory=True)
    for data in test_loader:
        print(data[0].shape)
        print(data[1].dtype)
