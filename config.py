
default_param_dicts = {
    "mnist": {
        "dataset": ["mnist"],
        "n_classes": [10],
        "n_clients": [10],
        "nc_per_client": [2],
        "n_client_perc": [None],
        "n_max_sam": [None],
        "c_ratio": [1.0],
        "net": ["LeNet"],
        "max_round": [200],
        "test_round": [1],
        "local_epochs": [2],
        "local_steps": [None],
        "batch_size": [64],
        "optimizer": ["SGD"],
        "lr": [0.05],
        "momentum": [0.9],
        "weight_decay": [1e-5],
        "max_grad_norm": [100.0],
        "cuda": [True],
    },
    "cifar10": {
        "dataset": ["cifar10"],
        "n_classes": [10],
        "n_clients": [100],
        "nc_per_client": [2],
        "n_client_perc": [None],
        "n_max_sam": [None],
        "c_ratio": [0.1],
        "net": ["TFCNN"],
        "max_round": [1000],
        "test_round": [10],
        "local_epochs": [2],
        "local_steps": [None],
        "batch_size": [64],
        "optimizer": ["SGD"],
        "lr": [0.03],
        "momentum": [0.9],
        "weight_decay": [1e-5],
        "max_grad_norm": [100.0],
        "cuda": [True],
    },
    "cifar100": {
        "dataset": ["cifar100"],
        "n_classes": [100],
        "n_clients": [100],
        "nc_per_client": [20],
        "n_client_perc": [None],
        "n_max_sam": [None],
        "c_ratio": [0.1],
        "net": ["TFCNN"],
        "max_round": [1000],
        "test_round": [10],
        "local_epochs": [2],
        "local_steps": [None],
        "batch_size": [64],
        "optimizer": ["SGD"],
        "lr": [0.03],
        "momentum": [0.9],
        "weight_decay": [1e-5],
        "max_grad_norm": [100.0],
        "cuda": [True],
    },
    "femnist": {
        "dataset": ["femnist"],
        "n_classes": [62],
        "n_clients": [None],
        "nc_per_client": [None],
        "n_client_perc": [None],
        "n_max_sam": [None],
        "c_ratio": [0.01],
        "net": ["FeMnistNet"],
        "max_round": [1000],
        "test_round": [10],
        "local_epochs": [2],
        "local_steps": [None],
        "batch_size": [10],
        "optimizer": ["SGD"],
        "lr": [4e-3],
        "momentum": [0.9],
        "weight_decay": [1e-5],
        "max_grad_norm": [100.0],
        "cuda": [True],
    },
    "shakespeare": {
        "dataset": ["shakespeare"],
        "n_vocab": [81],
        "n_classes": [81],
        "n_clients": [None],
        "nc_per_client": [None],
        "n_client_perc": [None],
        "n_max_sam": [None],
        "c_ratio": [0.01],
        "net": ["CharLSTM"],
        "max_round": [1000],
        "test_round": [10],
        "local_epochs": [None],
        "local_steps": [50],
        "batch_size": [10],
        "optimizer": ["SGD"],
        "lr": [1.47],
        "momentum": [0.9],
        "weight_decay": [1e-5],
        "max_grad_norm": [100.0],
        "cuda": [True],
    },
    "sa": {
        "dataset": ["sa"],
        "n_classes": [32],
        "n_clients": [89],
        "nc_per_client": [None],
        "n_client_perc": [70],
        "n_max_sam": [None],
        "c_ratio": [0.1],
        "net": ["SACNN"],
        "max_round": [1000],
        "test_round": [10],
        "local_epochs": [2],
        "local_steps": [None],
        "batch_size": [64],
        "optimizer": ["SGD"],
        "lr": [0.03],
        "momentum": [0.9],
        "weight_decay": [1e-5],
        "max_grad_norm": [100.0],
        "cuda": [True],
    },
}
