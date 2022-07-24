import os
import random
from collections import namedtuple

import torch

from paths import save_dir

from feddata import FedData
from fedps_auto import FedPSAuto

from ps_nets import AutoPSNet

from config import default_param_dicts
from utils import weights_init

torch.set_default_tensor_type(torch.FloatTensor)

# fedps_auto.py
# reference: http://www.lamda.nju.edu.cn/lixc/papers/FedPS-CoRR-Lixc.pdf
# Figure.10


def construct_model(args):
    model = AutoPSNet(
        net=args.net,
        way=args.way,
        n_classes=args.n_classes
    )
    print("***************************************")
    model.apply(weights_init)
    return model


def main_federated(para_dict):
    print(para_dict)
    param_names = para_dict.keys()
    Args = namedtuple("Args", param_names)
    args = Args(**para_dict)

    # DataSets
    feddata = FedData(
        dataset=args.dataset,
        n_clients=args.n_clients,
        nc_per_client=args.nc_per_client,
        n_client_perc=args.n_client_perc,
        n_max_sam=args.n_max_sam,
    )
    csets, gset = feddata.construct()
    feddata.print_info(csets, gset)

    # Model
    model = construct_model(args)
    print(model)
    print([name for name, _ in model.named_parameters()])
    n_params = sum([
        param.numel() for param in model.parameters()
    ])
    print("Total number of parameters : {}".format(n_params))

    if args.cuda:
        torch.backends.cudnn.benchmark = True

    # Train FedPS
    algo = FedPSAuto(
        csets=csets,
        gset=gset,
        model=model,
        args=args
    )
    algo.train()

    fpath = os.path.join(
        save_dir, args.fname
    )
    algo.save_logs(fpath)
    print(algo.logs)


def main_cifar10():
    # cifar100 datasets
    # dataset, n_clients, nc_per_client
    nets = ["VGG11"]
    ways = [
        "cs", "attn", "gs"
    ]

    pairs = [
        ("cifar10", 100, 5),
    ]

    for dataset, n_clients, nc_per_client in pairs:
        for net in nets:
            for way in ways:
                for lr in [0.05, 0.03]:
                    para_dict = {}
                    for k, vs in default_param_dicts[dataset].items():
                        para_dict[k] = random.choice(vs)

                    para_dict["dataset"] = dataset
                    para_dict["n_clients"] = n_clients
                    para_dict["nc_per_client"] = nc_per_client
                    para_dict["c_ratio"] = 0.1
                    para_dict["local_epochs"] = 2
                    para_dict["net"] = net
                    para_dict["way"] = way
                    para_dict["lr"] = lr
                    para_dict["fname"] = "fedps-auto-cifar10.log"

                    main_federated(para_dict)


def main_cifar100():
    # cifar100 datasets
    # dataset, n_clients, nc_per_client
    nets = ["VGG11"]
    ways = [
        "cs", "attn", "gs"
    ]

    pairs = [
        ("cifar100", 100, 20),
    ]

    for dataset, n_clients, nc_per_client in pairs:
        for net in nets:
            for way in ways:
                for lr in [0.05, 0.03]:
                    para_dict = {}
                    for k, vs in default_param_dicts[dataset].items():
                        para_dict[k] = random.choice(vs)

                    para_dict["dataset"] = dataset
                    para_dict["n_clients"] = n_clients
                    para_dict["nc_per_client"] = nc_per_client
                    para_dict["c_ratio"] = 0.1
                    para_dict["local_epochs"] = 2
                    para_dict["net"] = net
                    para_dict["way"] = way
                    para_dict["lr"] = lr
                    para_dict["test_round"] = 2
                    para_dict["fname"] = "fedps-auto-cifar100.log"

                    main_federated(para_dict)


def main_shake():
    ways = [
        "cs", "attn", "gs"
    ]

    dataset = "shakespeare"
    for c_ratio in [0.01]:
        for way in ways:
            for lr in [1.47, 1.0]:
                para_dict = {}
                for k, vs in default_param_dicts[dataset].items():
                    para_dict[k] = random.choice(vs)

                para_dict["dataset"] = dataset
                para_dict["c_ratio"] = c_ratio
                para_dict["local_steps"] = 50
                para_dict["batch_size"] = 50
                para_dict["way"] = way
                para_dict["lr"] = lr
                para_dict["fname"] = "fedps-auto-shake.log"

                main_federated(para_dict)


def main_femnist():
    ways = [
        "cs", "attn", "gs"
    ]

    dataset = "femnist"
    for c_ratio in [0.001]:
        for way in ways:
            for lr in [4e-3, 2e-3]:
                para_dict = {}
                for k, vs in default_param_dicts[dataset].items():
                    para_dict[k] = random.choice(vs)

                para_dict["dataset"] = dataset
                para_dict["c_ratio"] = c_ratio
                para_dict["local_epochs"] = 2
                para_dict["batch_size"] = 10
                para_dict["way"] = way
                para_dict["lr"] = lr
                para_dict["fname"] = "fedps-auto-femnist.log"

                main_federated(para_dict)


if __name__ == "__main__":
    datasets = [
        "cifar10", "cifar100",
        "shakespeare", "femnist",
    ]

    dataset = datasets[1]
    if dataset == "cifar10":
        main_cifar10()
    elif dataset == "cifar100":
        main_cifar100()
    elif dataset == "shakespeare":
        main_shake()
    elif dataset == "femnist":
        main_femnist()
