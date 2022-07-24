import os
import random
from collections import namedtuple

import torch

from paths import save_dir

from feddata import FedData
from fedps_inter import FedPSInter

from ps_nets import get_ps_net

from config import default_param_dicts

torch.set_default_tensor_type(torch.FloatTensor)


# fedps_inter.py
# reference: http://www.lamda.nju.edu.cn/lixc/papers/FedPS-CoRR-Lixc.pdf
# Figure.9
# Step 1: train SP2Net with split-layer=0, i.e., complete shared and private models
# Step 2: interpolate two models via inter_test()


def construct_model(args):
    model = get_ps_net(
        ps_type=args.ps_type,
        net=args.net,
        split_layer=args.split_layer,
        n_classes=args.n_classes
    )
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
    algo = FedPSInter(
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
    ps_types = ["SP2Net"]
    split_layers = [0]

    pairs = [
        ("cifar10", 100, 5),
    ]

    for dataset, n_clients, nc_per_client in pairs:
        for net in nets:
            for ps_type in ps_types:
                for split_layer in split_layers:
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
                        para_dict["ps_type"] = ps_type
                        para_dict["split_layer"] = split_layer
                        para_dict["lr"] = lr
                        para_dict["fname"] = "fedps-inter-cifar10.log"

                        main_federated(para_dict)


def main_cifar100():
    # cifar100 datasets
    # dataset, n_clients, nc_per_client
    nets = ["VGG11"]
    ps_types = ["SP2Net"]
    split_layers = [0]

    pairs = [
        ("cifar100", 100, 20),
    ]

    for dataset, n_clients, nc_per_client in pairs:
        for net in nets:
            for ps_type in ps_types:
                for split_layer in split_layers:
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
                        para_dict["ps_type"] = ps_type
                        para_dict["split_layer"] = split_layer
                        para_dict["lr"] = lr
                        para_dict["fname"] = "fedps-inter-cifar100.log"

                        main_federated(para_dict)


def main_shake():
    ps_types = ["SP2Net"]
    split_layers = [0]

    dataset = "shakespeare"
    for c_ratio in [0.01]:
        for ps_type in ps_types:
            for split_layer in split_layers:
                for lr in [1.47, 1.0]:
                    para_dict = {}
                    for k, vs in default_param_dicts[dataset].items():
                        para_dict[k] = random.choice(vs)

                    para_dict["dataset"] = dataset
                    para_dict["c_ratio"] = c_ratio
                    para_dict["local_steps"] = 50
                    para_dict["batch_size"] = 50
                    para_dict["ps_type"] = ps_type
                    para_dict["split_layer"] = split_layer
                    para_dict["lr"] = lr
                    para_dict["fname"] = "fedps-inter-shake.log"

                    main_federated(para_dict)


def main_femnist():
    ps_types = ["SP2Net"]
    split_layers = [0]

    dataset = "femnist"
    for c_ratio in [0.001]:
        for ps_type in ps_types:
            for split_layer in split_layers:
                for lr in [4e-3, 2e-3]:
                    para_dict = {}
                    for k, vs in default_param_dicts[dataset].items():
                        para_dict[k] = random.choice(vs)

                    para_dict["dataset"] = dataset
                    para_dict["c_ratio"] = c_ratio
                    para_dict["local_epochs"] = 2
                    para_dict["batch_size"] = 10
                    para_dict["ps_type"] = ps_type
                    para_dict["split_layer"] = split_layer
                    para_dict["lr"] = lr
                    para_dict["max_round"] = 200
                    para_dict["fname"] = "fedps-inter-femnist.log"

                    main_federated(para_dict)


def print_models():
    ps_types = ["PSNet", "SPNet", "PS2Net", "SP2Net"]
    split_layers = [0, 1]

    for ps_type in ps_types:
        for split_layer in split_layers:
            print("#" * 50)
            print(ps_type, split_layer)
            model = get_ps_net(
                ps_type=ps_type,
                net="TFCNN",
                split_layer=split_layer,
                n_classes=10
            )
            print(model)
            print("#" * 50)


if __name__ == "__main__":
    # print_models()

    datasets = [
        "cifar10", "cifar100", "shakespeare", "femnist",
    ]

    for dataset in datasets[3:]:
        if dataset == "cifar10":
            main_cifar10()
        elif dataset == "cifar100":
            main_cifar100()
        elif dataset == "shakespeare":
            main_shake()
        elif dataset == "femnist":
            main_femnist()
