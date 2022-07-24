import torch
from torch.utils.data import DataLoader


def construct_dataloaders(clients, csets, gset, args):
    train_loaders = {}
    test_loaders = {}
    glo_test_loader = None

    for client in clients:
        assert isinstance(csets[client], tuple), \
            "csets must be a tuple (train_set, test_set): {}".format(client)

        assert csets[client][1] is not None, \
            "local test set must not be None in client: {}".format(client)

        train_loader = DataLoader(
            csets[client][0],
            batch_size=args.batch_size,
            shuffle=True
        )
        train_loaders[client] = train_loader

        test_loader = DataLoader(
            csets[client][1],
            batch_size=args.batch_size * 50,
            shuffle=False
        )
        test_loaders[client] = test_loader

    assert gset is not None, \
        "global test set must not be None"

    glo_test_loader = DataLoader(
        gset,
        batch_size=args.batch_size,
        shuffle=False
    )

    return train_loaders, test_loaders, glo_test_loader


def construct_optimizer(model, lr, args):
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError("No such optimizer: {}".format(
            args.optimizer
        ))
    return optimizer
