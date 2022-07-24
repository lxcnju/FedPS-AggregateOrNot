import numpy as np

from collections import Counter

from mnist_data import load_mnist_data, MnistDataset
from cifar_data import load_cifar_data, CifarDataset
from shakespeare_data import load_shakespeare_data, ShakespeareDataset
from digits_data import load_digits_data, DigitsDataset
from femnist_data import load_femnist_data, FEMNISTDataset


np.random.seed(0)


class FedData():
    """ Federated Datasets: support different scenes and split ways
    params:
    @dataset: "mnist", "cifar10", "cifar100",
              "digits-five", "femnist",
              "sent140", "shakespeare"
    @split: "label", "user", None
        if split by "user", split each user to a client;
        if split by "label", split to n_clients with samples from several class
    @n_clients: int, None
        if split by "user", is Num.users;
        if split by "label", it is pre-defined;
    @nc_per_client: int, None
        number of classes per client, only for split="label";
    @n_client_perc: int, None
        number of clients per class, only for split="label" and dataset="sa";
    @n_max_sam: int, None
        max number of samples per client, for low-resource learning;
    @split_sent140_way: str
        the way to split sent140
    """

    def __init__(
        self,
        dataset="mnist",
        test_ratio=0.2,
        split=None,
        n_clients=None,
        nc_per_client=None,
        n_client_perc=None,
        n_max_sam=None,
    ):
        self.dataset = dataset
        self.test_ratio = test_ratio
        self.split = split
        self.n_clients = n_clients
        self.nc_per_client = nc_per_client
        self.n_client_perc = n_client_perc
        self.n_max_sam = n_max_sam

        self.label_dsets = [
            "mnist", "svhn", "mnistm", "usps", "syn",
            "cifar10", "cifar100",
            "sa"
        ]
        self.user_dsets = ["digits-five", "femnist", "shakespeare"]

        if dataset in self.label_dsets:
            self.split = "label"

            assert (n_clients is not None), \
                "{} needs pre-defined n_clients".format(dataset)

            if dataset == "sa":
                assert (n_client_perc is not None), \
                    "{} needs pre-defined n_client_perc".format(dataset)
            else:
                assert (nc_per_client is not None), \
                    "{} needs pre-defined nc_per_client".format(dataset)

        if dataset in self.user_dsets:
            self.split = "user"

    def split_by_label(self, xs, ys):
        """ split data into N clients, each client has C classes
        params:
        @xs: numpy.array, shape=(N, ...)
        @ys: numpy.array, shape=(N, ), only for classes
        return:
        @clients_data, a dict like {
            client: {
                "train_xs":,
                "train_ys":,
                "test_xs":,
                "test_ys":
            }
        }
        """
        # unique classes
        uni_classes = sorted(np.unique(ys))
        seq_classes = []
        for _ in range(self.n_clients):
            np.random.shuffle(uni_classes)
            seq_classes.extend(list(uni_classes))

        # each class at least assigned to a client
        assert (self.nc_per_client * self.n_clients >= len(uni_classes)), \
            "Each class as least assigned to a client"

        # assign classes to each client
        client_classes = {}
        for k, client in enumerate(range(self.n_clients)):
            client_classes[client] = seq_classes[
                k * self.nc_per_client: (k + 1) * self.nc_per_client
            ]

        # for a class, how many clients have it
        classes = []
        for client in client_classes.keys():
            classes.extend(client_classes[client])

        classes_cnt = dict(Counter(classes))

        # shuffle xs, and ys
        n_samples = xs.shape[0]
        inds = np.random.permutation(n_samples)
        xs = xs[inds]
        ys = ys[inds]

        # assign classes to each client
        clients_data = {}
        for client in client_classes.keys():
            clients_data[client] = {
                "xs": [],
                "ys": []
            }

        # split data by classes
        for c in uni_classes:
            cinds = np.argwhere(ys == c).reshape(-1)
            c_xs = xs[cinds]
            c_ys = ys[cinds]

            # assign class data uniformly to each client
            t = 0
            for client, client_cs in client_classes.items():
                if c in client_cs:
                    ind1 = t * int(len(c_xs) / classes_cnt[c])
                    ind2 = (t + 1) * int(len(c_xs) / classes_cnt[c])
                    clients_data[client]["xs"].append(c_xs[ind1:ind2])
                    clients_data[client]["ys"].append(c_ys[ind1:ind2])
                    t += 1
            assert (t == classes_cnt[c]), \
                "Error, t != classes_cnt[c]"

        # shuffle data and limit maximum number
        for client, values in clients_data.items():
            client_xs = np.concatenate(values["xs"], axis=0)
            client_ys = np.concatenate(values["ys"], axis=0)

            inds = np.random.permutation(client_xs.shape[0])
            client_xs = client_xs[inds]
            client_ys = client_ys[inds]

            # filter small corpus
            if len(client_xs) < 5:
                continue

            # split train and test
            n_test = max(int(self.test_ratio * len(client_xs)), 1)

            # max train samples
            if self.n_max_sam is None:
                n_end = None
            else:
                n_end = self.n_max_sam + n_test

            clients_data[client] = {
                "train_xs": client_xs[n_test:n_end],
                "train_ys": client_ys[n_test:n_end],
                "test_xs": client_xs[:n_test],
                "test_ys": client_ys[:n_test],
            }

        return clients_data

    def split_by_label_sa(self, xs, ys):
        infos = {}
        for k in range(self.n_clients):
            infos["client-{}".format(k)] = {
                "xs": [],
                "ys": []
            }

        clients = list(infos.keys())

        for c in np.unique(ys):
            cinds = np.argwhere(ys == c).reshape(-1)

            c_xs = xs[cinds]
            c_ys = ys[cinds]

            nc0 = max(self.n_client_perc - 5, 1)
            nc1 = min(self.n_client_perc + 5, self.n_clients)
            nc = np.random.choice(range(nc0, nc1))
            sub_clients = np.random.choice(clients, nc)

            n_perc = int(len(c_xs) / nc)

            for k, client in enumerate(sub_clients):
                i = k * n_perc
                j = (k + 1) * n_perc
                infos[client]["xs"].append(c_xs[i:j])
                infos[client]["ys"].append(c_ys[i:j])

        ft_infos = {}
        for client, cdata in infos.items():
            if len(cdata["xs"]) > 0:
                ft_infos[client] = {
                    "xs": np.concatenate(cdata["xs"], axis=0),
                    "ys": np.concatenate(cdata["ys"], axis=0),
                }

        clients_data = {}
        for client, cdata in ft_infos.items():
            inds = np.random.permutation(cdata["xs"].shape[0])
            client_xs = cdata["xs"][inds]
            client_ys = cdata["ys"][inds]

            # filter small corpus
            if len(client_xs) < 5:
                continue

            # split train and test
            n_test = max(int(self.test_ratio * len(client_xs)), 1)

            # max train samples
            if self.n_max_sam is None:
                n_end = None
            else:
                n_end = self.n_max_sam + n_test

            clients_data[client] = {
                "train_xs": client_xs[n_test:n_end],
                "train_ys": client_ys[n_test:n_end],
                "test_xs": client_xs[:n_test],
                "test_ys": client_ys[:n_test],
            }

        return clients_data

    def split_shakespeare(self, users_data):
        clients_data = {}

        for client, info in users_data.items():
            n_test = max(int(self.test_ratio * len(info["xs"])), 1)

            # max train samples
            if self.n_max_sam is None:
                n_end = None
            else:
                n_end = self.n_max_sam + n_test

            clients_data[client] = {
                "train_xs": info["xs"][n_test:n_end],
                "train_ys": info["ys"][n_test:n_end],
                "test_xs": info["xs"][:n_test],
                "test_ys": info["ys"][:n_test],
            }
        return clients_data

    def split_femnist(self, users_data):
        clients_data = {}

        for client, info in users_data.items():
            n_test = max(int(self.test_ratio * len(info["xs"])), 1)
            inds = np.random.permutation(info["xs"].shape[0])
            client_xs = info["xs"][inds]
            client_ys = info["ys"][inds]

            # max train samples
            if self.n_max_sam is None:
                n_end = None
            else:
                n_end = self.n_max_sam + n_test

            clients_data[client] = {
                "train_xs": client_xs[n_test:n_end],
                "train_ys": client_ys[n_test:n_end],
                "test_xs": client_xs[:n_test],
                "test_ys": client_ys[:n_test],
            }
        return clients_data

    def construct_datasets(self, clients_data, Dataset):
        """
        params:
        @clients_data, {
            client: {
                "train_xs":,
                "train_ys":,
                "test_xs":,
                "test_ys":
            }
        }
        @Dataset: torch.utils.data.Dataset type
        return: client train and test Datasets and global test Dataset
        @csets: {
            client: (train_set, test_set)
        }
        @gset: data.Dataset
        """
        csets = {}
        glo_test_xs = []
        glo_test_ys = []
        for client, cdata in clients_data.items():
            train_set = Dataset(
                cdata["train_xs"], cdata["train_ys"], is_train=True
            )
            test_set = Dataset(
                cdata["test_xs"], cdata["test_ys"], is_train=False
            )
            glo_test_xs.append(cdata["test_xs"])
            glo_test_ys.append(cdata["test_ys"])
            csets[client] = (train_set, test_set)

        glo_test_xs = np.concatenate(glo_test_xs, axis=0)
        glo_test_ys = np.concatenate(glo_test_ys, axis=0)
        gset = Dataset(glo_test_xs, glo_test_ys, is_train=False)
        return csets, gset

    def construct(self):
        """ load raw data
        """
        if self.dataset == "mnist":
            xs, ys = load_mnist_data("mnist", combine=True)
            clients_data = self.split_by_label(xs, ys)
            csets, gset = self.construct_datasets(
                clients_data, MnistDataset
            )
        elif self.dataset in ["mnistm", "svhn", "usps", "syn"]:
            xs, ys = load_digits_data(
                self.dataset, combine=True
            )
            clients_data = self.split_by_label(xs, ys)
            csets, gset = self.construct_datasets(
                clients_data, DigitsDataset
            )
        elif self.dataset in ["cifar10", "cifar100"]:
            xs, ys = load_cifar_data(
                self.dataset, combine=True
            )
            clients_data = self.split_by_label(xs, ys)
            csets, gset = self.construct_datasets(
                clients_data, CifarDataset
            )
        elif self.dataset == "shakespeare":
            users_data = load_shakespeare_data()
            clients_data = self.split_shakespeare(users_data)
            csets, gset = self.construct_datasets(
                clients_data, ShakespeareDataset
            )
        elif self.dataset == "digits-five":
            clients_data = {}
            domains = ["mnist", "mnistm", "usps", "svhn", "syn"]
            for domain in domains:
                train_xs, train_ys, test_xs, test_ys = load_digits_data(
                    domain, combine=False
                )

                inds = np.random.permutation(train_xs.shape[0])
                train_xs = train_xs[inds]
                train_ys = train_ys[inds]

                clients_data[domain] = {
                    "train_xs": train_xs[0:self.n_max_sam],
                    "train_ys": train_ys[0:self.n_max_sam],
                    "test_xs": test_xs,
                    "test_ys": test_ys,
                }
            csets, gset = self.construct_datasets(
                clients_data, DigitsDataset
            )
        elif self.dataset == "femnist":
            users_data = load_femnist_data()
            clients_data = self.split_femnist(users_data)
            csets, gset = self.construct_datasets(
                clients_data, FEMNISTDataset
            )
        else:
            raise ValueError("No such dataset: {}".format(self.dataset))

        return csets, gset

    def print_info(self, csets, gset, max_cnt=5):
        """ print information
        """
        print("#" * 50)
        cnt = 0
        print("Dataset:{}".format(self.dataset))
        print("N clients:{}".format(len(csets)))

        for client, (cset1, cset2) in csets.items():
            print("Information of Client {}:".format(client))
            print(
                "Local Train Set: ", cset1.xs.shape,
                cset1.xs.max(), cset1.xs.min(), Counter(cset1.ys)
            )
            print(
                "Local Test Set: ", cset2.xs.shape,
                cset2.xs.max(), cset2.xs.min(), Counter(cset2.ys)
            )

            if cnt >= max_cnt:
                break
            cnt += 1

        print(
            "Global Test Set: ", gset.xs.shape,
            gset.xs.max(), gset.xs.min(), Counter(gset.ys)
        )
        print("#" * 50)


if __name__ == "__main__":
    for dataset in ["mnist", "svhn", "cifar10", "cifar100"]:
        for n_max_sam in [500, None]:
            for n_clients in [100]:
                for nc_per_client in [5]:
                    print("#" * 50)
                    feddata = FedData(
                        dataset=dataset,
                        n_clients=n_clients,
                        nc_per_client=nc_per_client,
                        n_max_sam=n_max_sam
                    )
                    csets, gset = feddata.construct()
                    feddata.print_info(csets, gset, max_cnt=5)

    for dataset in ["sa"]:
        for n_max_sam in [500, None]:
            for n_clients in [100]:
                for n_client_perc in [50]:
                    print("#" * 50)
                    feddata = FedData(
                        dataset=dataset,
                        n_clients=n_clients,
                        n_client_perc=n_client_perc,
                        n_max_sam=n_max_sam
                    )
                    csets, gset = feddata.construct()
                    feddata.print_info(csets, gset, max_cnt=5)

    for n_max_sam in [50, 500, None]:
        feddata = FedData(
            dataset="digits-five",
            n_max_sam=n_max_sam
        )
        print("#" * 50)
        csets, gset = feddata.construct()
        feddata.print_info(csets, gset, max_cnt=5)

    for n_max_sam in [50, 500, None]:
        feddata = FedData(
            dataset="shakespeare",
            n_max_sam=n_max_sam
        )
        print("#" * 50)
        csets, gset = feddata.construct()
        feddata.print_info(csets, gset, max_cnt=5)
