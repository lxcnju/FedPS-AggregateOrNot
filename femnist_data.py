import os
import copy
import numpy as np
# from shutil import copyfile
from PIL import Image

import torch
from torch.utils import data
from torchvision import transforms

from utils import load_pickle

from paths import femnist_fdirs


def load_femnist_data():
    users_data = {}

    for femnist_fdir in femnist_fdirs:
        fnames = os.listdir(femnist_fdir)
        fnames = [fname for fname in fnames if fname.endswith(".pkl")]
        clients = [fname.split(".")[0] for fname in fnames]

        for client, fname in zip(clients, fnames):
            fpath = os.path.join(femnist_fdir, fname)
            (xs, ys) = load_pickle(fpath)
            users_data[client] = {
                "xs": xs,
                "ys": ys
            }
    return users_data


class FEMNISTDataset(data.Dataset):
    def __init__(self, xs, ys, is_train=True):
        self.xs = copy.deepcopy(xs)
        self.ys = copy.deepcopy(ys)
        self.is_train = is_train

        if is_train is True:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize([28, 28]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize([28, 28]),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, index):
        raw_img = self.xs[index].reshape((1, 28, 28))
        label = self.ys[index]

        # transforms.ToPILImage need (H, W, C) np.uint8 input
        img = raw_img.transpose(1, 2, 0).astype(np.uint8)

        # return (C, H, W) tensor
        img = self.transform(img)

        label = torch.LongTensor([label])[0]
        return img, label


"""
def process_femnist():
    fdir_u = r"C:\Workspace\work\datasets\femnist_by_user"
    fdir_c = r"C:\Workspace\work\datasets\femnist_by_class"
    fdir = r"C:\Workspace\work\datasets\femnist"

    fdict = {}
    for (d1, _, fnames) in os.walk(fdir_c):
        fnames = [fname for fname in fnames if fname.endswith(".mit")]
        if len(fnames) <= 0:
            continue

        for fname in fnames:
            with open(os.path.join(d1, fname)) as fr:
                for line in fr:
                    line = line.strip()
                    parts = line.split()
                    if len(parts) == 2:
                        norm_key = "_".join(parts[1].split("_")[0:-1])
                        aft_key = parts[1].split("_")[-1]
                        if "000" in aft_key and len(aft_key) == 10:
                            norm_key += "_" + aft_key[1:]
                        else:
                            norm_key += "_" + aft_key

                        fdict[norm_key] = d1[-2:]

    print(len(fdict))

    all_classes = list(sorted(np.unique(
        [c for _, c in fdict.items()]
    )))
    print(all_classes)

    data = {}
    cnt = 0
    for subdir1 in os.listdir(fdir_u):
        for subdir2 in os.listdir(os.path.join(fdir_u, subdir1)):
            user = subdir2.split("_")[0]

            if user not in data:
                data[user] = {}

            cdir = os.path.join(fdir_u, subdir1, subdir2)
            for (d1, _, fnames) in os.walk(cdir):
                for fname in fnames:
                    fpath = os.path.join(d1, fname)
                    key = "/".join(str(fpath).split("\\")[-3:])

                    try:
                        label = fdict[key]
                    except Exception:
                        cnt += 1
                        print(fpath)

                    if label in data[user]:
                        data[user][label].append(fpath)
                    else:
                        data[user][label] = [fpath]

    print(cnt)
    print(len(data))
    print([len(udata) for _, udata in data.items()])

    for user in data.keys():
        try:
            os.mkdir(os.path.join(fdir, user))
        except Exception:
            pass

        for c in all_classes:
            to_dir = os.path.join(fdir, user, c)
            if os.path.exists(to_dir):
                continue
            else:
                os.mkdir(to_dir)

        for label, paths in data[user].items():
            subdir = os.path.join(fdir, user, label)

            for k, path in enumerate(paths):
                name = "{}-{}-{:0>6d}.png".format(user, label, k + 1)
                copyfile(path, os.path.join(subdir, name))

        print("User {} done!".format(user))
"""

"""
def load_femnist_data():
    data = {}
    class2int = None
    for udir in os.listdir(femnist_fdir):
        fpaths = []
        labels = []
        all_labels = list(sorted(os.listdir(os.path.join(femnist_fdir, udir))))
        if class2int is None:
            class2int = {label: i for i, label in enumerate(all_labels)}

        for cdir in os.listdir(os.path.join(femnist_fdir, udir)):
            for fname in os.listdir(os.path.join(femnist_fdir, udir, cdir)):
                fpaths.append(os.path.join(femnist_fdir, udir, cdir, fname))
                labels.append(class2int[cdir])

        fpaths = np.array(fpaths)
        labels = np.array(labels)
        inds = np.random.permutation(len(fpaths))
        fpaths = fpaths[inds]
        labels = labels[inds]

        n_train = int(0.8 * len(inds))

        if len(fpaths) < 5:
            continue

        train_fpaths, train_labels = fpaths[0:n_train], labels[0:n_train]
        test_fpaths, test_labels = fpaths[n_train:], labels[n_train:]

        data[udir] = {
            "train_xs": train_fpaths,
            "train_ys": train_labels,
            "test_xs": test_fpaths,
            "test_ys": test_labels,
        }

    return data
"""

"""
def load_femnist_data():
    data = {}
    fpaths = [fp for fp in os.listdir(femnist_fdir) if fp.endswith(".png")]

    fpaths = np.array(fpaths)

    users = [fp.split("-")[0] for fp in fpaths]
    labels = [fp.split("-")[1] for fp in fpaths]

    users = np.array(users)
    labels = np.array(labels)

    uni_users = np.unique(users)
    uni_labels = np.unique(labels)
    print(len(uni_users), len(uni_labels))

    uni_users = list(sorted(uni_users))
    user2int = {user: i for i, user in enumerate(uni_users)}
    users = [user2int[u] for u in users]
    uni_users = list(sorted(np.unique(users)))

    uni_labels = list(sorted(uni_labels))
    label2int = {label: i for i, label in enumerate(uni_labels)}
    labels = [label2int[c] for c in labels]
    uni_labels = list(sorted(np.unique(labels)))

    data = {}
    for uid in uni_users:
        uinds = np.argwhere(users == uid).reshape(-1)
        u_fpaths = [
            os.path.join(femnist_fdir, fp) for fp in fpaths[uinds]
        ]
        u_labels = [
            label2int[fp.split("-")[1]] for fp in fpaths[uinds]
        ]

        u_fpaths = np.array(u_fpaths)
        u_labels = np.array(u_labels)
        u_inds = np.random.permutation(len(u_fpaths))
        u_fpaths = u_fpaths[u_inds]
        u_labels = u_labels[u_inds]

        n_train = int(0.8 * len(u_inds))

        if len(u_fpaths) < 5:
            continue

        train_fpaths, train_labels = u_fpaths[0:n_train], u_labels[0:n_train]
        test_fpaths, test_labels = u_fpaths[n_train:], u_labels[n_train:]

        data[uid] = {
            "train_xs": train_fpaths,
            "train_ys": train_labels,
            "test_xs": test_fpaths,
            "test_ys": test_labels,
        }

    return data
"""

