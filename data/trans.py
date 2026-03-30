# -*- coding: utf-8 -*-

from collections.abc import Sequence
import numpy as np

class Base(object):
    def sample(self, *shape):
        return shape

    def tf(self, img, k=0):
        return img

    def __call__(self, img, dim=3, reuse=False):
        if not reuse:
            im = img if isinstance(img, np.ndarray) else img[0]
            shape = im.shape[1:dim+1]
            self.sample(*shape)

        if isinstance(img, Sequence):
            return [self.tf(x, k) for k, x in enumerate(img)]

        return self.tf(img)

    def __str__(self):
        return 'Identity()'

class Seg_norm(Base):
    """
    Test for comparative experiments
    """
    def __init__(self, ):
        self.seg_table = np.array([0, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 61, 62,
               63, 64, 65, 66, 67, 68, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 101, 102, 121, 122, 161, 162,
               163, 164, 165, 166])
    def tf(self, img, k=0):
        if k == 0:
            return img
        img_out = np.zeros_like(img)
        for i in range(len(self.seg_table)):
            img_out[img == self.seg_table[i]] = i
        return img_out

class Simple_Seg_norm(Base):
    """
    For box plot visualization and analysis
    """
    def __init__(self, ):
        self.seg_table = np.array([0, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 61, 62,
               63, 64, 65, 66, 67, 68, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 101, 102, 121, 122, 161, 162,
               163, 164, 165, 166]) # Label values corresponding to anatomical structure labels
        self.target_table = [[1, 2, 3, 4, 5, 6, 9, 10, 11, 12], [17, 18], [25, 26, 27, 28, 29, 30], [33, 34, 35, 36, 37, 38], [43, 44], [51, 52], [53, 54]] # Index corresponding to the seg_table
        self._build_mapping_table()

    def _build_mapping_table(self):
        label_mapping = {}
        for group_id, indices in enumerate(self.target_table):
            for idx in indices:
                if idx < len(self.seg_table):
                    original_label = self.seg_table[idx]
                    label_mapping[original_label] = group_id + 1
        if not label_mapping:
            self.lookup = np.array([0], dtype=np.int32)
            return
        max_label = self.seg_table.max().item()
        self.lookup = np.zeros(max_label + 1, dtype=np.int32)
        for original_label, new_label in label_mapping.items():
            self.lookup[original_label] = new_label

    def tf(self, img, k=0):
        if k == 0:
            return img
        img_out = np.zeros_like(img)
        valid_mask = (img >= 0) & (img <= len(self.lookup) - 1)
        img_out[valid_mask] = self.lookup[img[valid_mask]]
        return img_out

class NumpyType(Base):
    def __init__(self, types, num=-1):
        self.types = types # ('float32', 'int64')
        self.num = num

    def tf(self, img, k=0):
        if self.num > 0 and k >= self.num:
            return img
        # make this work with both Tensor and Numpy
        return img.astype(self.types[k])

    def __str__(self):
        s = ', '.join([str(s) for s in self.types])
        return 'NumpyType(({}))'.format(s)

