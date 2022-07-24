import numpy as np

from mnist_data import load_mnist_data, MnistDataset
from cifar_data import load_cifar_data, CifarDataset
from shakespeare_data import load_shakespeare_data, ShakespeareDataset
from digits_data import load_digits_data, DigitsDataset
from femnist_data import load_femnist_data, FEMNISTDataset


class Data():
    """ Classify Dataset
    """

    def __init__(
        self,
        dataset="mnist",
    ):
        self.dataset = dataset

    def construct(self):
        """ load raw data
        """
        if self.dataset == "mnist":
            train_xs, train_ys, test_xs, test_ys = load_mnist_data(
                "mnist", combine=False
            )
            train_set = MnistDataset(train_xs, train_ys)
            test_set = MnistDataset(test_xs, test_ys)
        elif self.dataset in ["mnistm", "svhn", "usps", "syn"]:
            train_xs, train_ys, test_xs, test_ys = load_digits_data(
                self.dataset, combine=False
            )
            train_set = DigitsDataset(train_xs, train_ys)
            test_set = DigitsDataset(test_xs, test_ys)
        elif self.dataset in ["cifar10", "cifar100"]:
            train_xs, train_ys, test_xs, test_ys = load_cifar_data(
                self.dataset, combine=False
            )
            train_set = CifarDataset(train_xs, train_ys, is_train=True)
            test_set = CifarDataset(test_xs, test_ys, is_train=False)
        elif self.dataset == "shakespeare":
            users_data = load_shakespeare_data()

            train_xs = []
            train_ys = []
            test_xs = []
            test_ys = []
            for client, info in users_data.items():
                n_train = int(0.8 * len(info["xs"]))
                train_xs.append(info["xs"][0:n_train])
                train_ys.append(info["ys"][0:n_train])
                test_xs.append(info["xs"][n_train:])
                test_ys.append(info["ys"][n_train:])

            train_xs = np.concatenate(train_xs, axis=0)
            train_ys = np.concatenate(train_ys, axis=0)
            test_xs = np.concatenate(test_xs, axis=0)
            test_ys = np.concatenate(test_ys, axis=0)

            train_set = ShakespeareDataset(train_xs, train_ys)
            test_set = ShakespeareDataset(test_xs, test_ys)
        elif self.dataset == "digits-five":
            train_xs = []
            train_ys = []
            test_xs = []
            test_ys = []

            domains = ["mnist", "mnistm", "usps", "svhn", "syn"]
            for domain in domains:
                tr_xs, tr_ys, te_xs, te_ys = load_digits_data(
                    domain, combine=False
                )
                train_xs.append(tr_xs)
                train_ys.append(tr_ys)
                test_xs.append(te_xs)
                test_ys.append(te_ys)

            train_xs = np.concatenate(train_xs, axis=0)
            train_ys = np.concatenate(train_ys, axis=0)
            test_xs = np.concatenate(test_xs, axis=0)
            test_ys = np.concatenate(test_ys, axis=0)

            train_set = DigitsDataset(train_xs, train_ys)
            test_set = DigitsDataset(test_xs, test_ys)
        elif self.dataset == "femnist":
            clients_data = load_femnist_data()
            csets = {}
            for client, info in clients_data.items():
                csets[client] = (
                    FEMNISTDataset(
                        info["train_xs"][0:self.n_max_sam],
                        info["train_ys"][0:self.n_max_sam],
                        is_train=True,
                    ),
                    FEMNISTDataset(
                        info["test_xs"],
                        info["test_ys"],
                        is_train=False,
                    ),
                )
        else:
            raise ValueError("No such dataset: {}".format(self.dataset))

        return train_set, test_set


if __name__ == "__main__":
    datasets = [
        "mnist", "mnistm", "svhn", "syn", "usps",
        "cifar10", "cifar100",
        "sa",
        "shakespeare",
        "digits-five"
    ]

    for dset in datasets:
        data = Data(dset)
        train_set, test_set = data.construct()
        print(train_set.xs.shape, train_set.ys.shape)
        print(test_set.xs.shape, test_set.ys.shape)
