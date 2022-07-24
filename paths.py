import os

data_dir = r"C:\Workspace\work\datasets"
cur_dir = "./"

digits_fdir = os.path.join(data_dir, "Digits-five")
cifar_fdir = os.path.join(data_dir, "Cifar")
femnist_fdirs = [
    os.path.join(data_dir, "femnist_pkls", "part1"),
    os.path.join(data_dir, "femnist_pkls", "part2"),
    os.path.join(data_dir, "femnist_pkls", "part3"),
    os.path.join(data_dir, "femnist_pkls", "part4"),
    os.path.join(data_dir, "femnist_pkls", "part5"),
]
shakespeare_fdir = os.path.join(data_dir, "Shakespeare")
sent140_fdir = os.path.join(data_dir, "Sent140")

save_dir = os.path.join(cur_dir, "logs")

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

sent140_raw_fpath = os.path.join(sent140_fdir, "Sent140.json")
sent140_fpath = os.path.join(sent140_fdir, "Sent140.pkl")

shakespeare_fpath = os.path.join(shakespeare_fdir, "Shakespeare.json")

cifar_fpaths = {
    "cifar10": {
        "train_fpaths": [
            os.path.join(cifar_fdir, "cifar10-train-part1.pkl"),
            os.path.join(cifar_fdir, "cifar10-train-part2.pkl"),
        ],
        "test_fpath": os.path.join(cifar_fdir, "cifar10-test.pkl")
    },
    "cifar100": {
        "train_fpaths": [
            os.path.join(cifar_fdir, "cifar100-train-part1.pkl"),
            os.path.join(cifar_fdir, "cifar100-train-part2.pkl"),
        ],
        "test_fpath": os.path.join(cifar_fdir, "cifar100-test.pkl")
    },
}
