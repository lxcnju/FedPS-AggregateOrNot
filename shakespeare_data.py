import copy
import numpy as np

import torch
from torch.utils import data

from paths import shakespeare_fpath

from utils import load_json


def load_shakespeare_data():
    def slice_index_(dset):
        xs, ys = dset["x"], dset["y"]
        xs = [int(x) for x in xs]
        ys = [int(y) for y in ys]
        n_ys = len(ys)

        slice_xs = []
        slice_ys = []

        for i in range(n_ys):
            slice_xs.append(xs[i:i + 80])
            slice_ys.append(ys[i])
            assert xs[i + 80] == ys[i]

        slice_xs = np.array(slice_xs)
        slice_ys = np.array(slice_ys)

        slice_dset = {
            "xs": slice_xs,
            "ys": slice_ys
        }
        return slice_dset

    data = load_json(shakespeare_fpath)
    for key, dset in data["user_data"].items():
        data["user_data"][key] = slice_index_(dset)

    users_data = data["user_data"]
    return users_data


class ShakespeareDataset(data.Dataset):
    def __init__(self, xs, ys, is_train=None):
        self.xs = copy.deepcopy(xs)
        self.ys = copy.deepcopy(ys)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, index):
        x = self.xs[index]
        y = self.ys[index]

        x = torch.LongTensor(x)
        y = torch.LongTensor([y])[0]
        return x, y
