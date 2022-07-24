import os
import random
from collections import namedtuple

import torch

from paths import save_dir

from feddata import FedData
from fedavg import FedAvg

from basic_nets import get_basic_net

from config import default_param_dicts

torch.set_default_tensor_type(torch.FloatTensor)


def construct_model(args):
    model = get_basic_net(args.net, args.n_classes)
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
        model = model.cuda()

    # Train FedAvg
    algo = FedAvg(
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
    pairs = [
        ("cifar10", 100, 5),
        ("cifar10", 100, 2),
        ("cifar10", 500, 5),
        ("cifar10", 500, 2),
    ]

    for dataset, n_clients, nc_per_client in pairs:
        for net in ["TFCNN", "VGG11"]:
            for c_ratio in [0.1, 0.5, 1.0]:
                for local_epochs in [1, 2, 5, 10]:
                    para_dict = {}
                    for k, vs in default_param_dicts[dataset].items():
                        para_dict[k] = random.choice(vs)

                    para_dict["dataset"] = dataset
                    para_dict["n_clients"] = n_clients
                    para_dict["nc_per_client"] = nc_per_client
                    para_dict["c_ratio"] = c_ratio
                    para_dict["local_epochs"] = local_epochs
                    para_dict["net"] = net
                    para_dict["fname"] = "fedavg-cifar10.log"

                    main_federated(para_dict)


def main_cifar100():
    # cifar100 datasets
    # dataset, n_clients, nc_per_client
    pairs = [
        ("cifar100", 100, 20),
        ("cifar100", 100, 10),
        ("cifar100", 500, 20),
        ("cifar100", 500, 10),
    ]

    for dataset, n_clients, nc_per_client in pairs:
        for net in ["TFCNN", "VGG11"]:
            for c_ratio in [0.1, 0.5, 1.0]:
                for local_epochs in [1, 2, 5, 10]:
                    para_dict = {}
                    for k, vs in default_param_dicts[dataset].items():
                        para_dict[k] = random.choice(vs)

                    para_dict["dataset"] = dataset
                    para_dict["n_clients"] = n_clients
                    para_dict["nc_per_client"] = nc_per_client
                    para_dict["c_ratio"] = c_ratio
                    para_dict["local_epochs"] = local_epochs
                    para_dict["net"] = net
                    para_dict["fname"] = "fedavg-cifar100.log"

                    main_federated(para_dict)


def main_shake():
    dataset = "shakespeare"
    for c_ratio in [0.01, 0.05, 0.1]:
        for local_steps in [50, 250]:
            for batch_size in [10, 50]:
                para_dict = {}
                for k, vs in default_param_dicts[dataset].items():
                    para_dict[k] = random.choice(vs)

                para_dict["dataset"] = dataset
                para_dict["c_ratio"] = c_ratio
                para_dict["local_steps"] = local_steps
                para_dict["batch_size"] = batch_size
                para_dict["fname"] = "fedavg-shake.log"

                main_federated(para_dict)


def main_femnist():
    dataset = "femnist"
    for c_ratio in [0.001, 0.005, 0.01]:
        for local_epochs in [1, 2, 5, 10]:
            for batch_size in [10, 50]:
                para_dict = {}
                for k, vs in default_param_dicts[dataset].items():
                    para_dict[k] = random.choice(vs)

                para_dict["dataset"] = dataset
                para_dict["c_ratio"] = c_ratio
                para_dict["local_epochs"] = 2
                para_dict["batch_size"] = 10
                para_dict["fname"] = "fedavg-femnist.log"

                main_federated(para_dict)



if __name__ == "__main__":
    datasets = ["cifar10", "cifar100", "shakespeare", "femnist"]

    dataset = datasets[3]
    if dataset == "cifar10":
        main_cifar10()
    elif dataset == "cifar100":
        main_cifar100()
    elif dataset == "shakespeare":
        main_shake()
    elif dataset == "femnist":
        main_femnist()

    