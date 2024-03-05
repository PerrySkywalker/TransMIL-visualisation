from torch.utils.data import Dataset
import h5py
import torch
import os
import numpy as np
class Attn_Dateset(Dataset):
    def __init__(self, h5_path, thumbnail_path):
        super().__init__()
        self.h5_dir = h5_path
        self.h5_names = os.listdir(h5_path)
        self.thumbnail_path = thumbnail_path
    def __getitem__(self, index):
        f = h5py.File(self.h5_dir + self.h5_names[index], 'r')
        coords = np.array(f["coords"])
        feature = torch.tensor(f["features"])
        f.close
        return coords, feature, self.thumbnail_path + self.h5_names[index][:-3] + '.jpg'

    def __len__(self):  
        return len(self.h5_names)