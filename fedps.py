import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import Averager
from utils import count_acc
from utils import append_to_logs
from utils import format_logs

from tools import construct_dataloaders
from tools import construct_optimizer

# fedps.py
# reference: http://www.lamda.nju.edu.cn/lixc/papers/FedPS-CoRR-Lixc.pdf
# Figure.5 & Figure.6 & Figure.7 & Figure.8


class FedPS():
    def __init__(
        self,
        csets,
        gset,
        model,
        args
    ):
        self.csets = csets
        self.gset = gset
        self.model = model
        self.args = args

        self.clients = list(csets.keys())
        self.n_client = len(self.clients)

        # copy private layers for each client
        self.private_models = {}
        for client in self.clients:
            self.private_models[client] = copy.deepcopy(model.private_layers)

        # to cuda
        if self.args.cuda is True:
            self.model = self.model.cuda()

        # construct dataloaders
        self.train_loaders, self.test_loaders, self.glo_test_loader = \
            construct_dataloaders(
                self.clients, self.csets, self.gset, self.args
            )

        self.logs = {
            "ROUNDS": [],
            "LOSSES": [],
            "GLO_TACCS": [],
            "LOCAL_TACCS": [],
        }

    def train(self):
        # Training
        for r in range(1, self.args.max_round + 1):
            n_sam_clients = int(self.args.c_ratio * len(self.clients))
            sam_clients = np.random.choice(
                self.clients, n_sam_clients, replace=False
            )

            local_models = {}

            avg_loss = Averager()
            all_per_accs = []
            for client in sam_clients:
                local_model = copy.deepcopy(self.model)

                # to cuda
                if self.args.cuda is True:
                    self.private_models[client].cuda()

                state_dict = {
                    "private_layers.{}".format(k): v for k, v in
                    self.private_models[client].state_dict().items()
                }

                local_model.load_state_dict(state_dict, strict=False)

                local_model, per_accs, loss = self.update_local(
                    r=r,
                    model=local_model,
                    train_loader=self.train_loaders[client],
                    test_loader=self.test_loaders[client],
                )

                local_models[client] = copy.deepcopy(local_model)

                # update local model
                state_dict = local_model.private_layers.state_dict()
                self.private_models[client].load_state_dict(
                    state_dict, strict=True
                )

                # to cpu
                if self.args.cuda is True:
                    self.private_models[client].to("cpu")

                avg_loss.add(loss)
                all_per_accs.append(per_accs)

            train_loss = avg_loss.item()
            per_accs = list(np.array(all_per_accs).mean(axis=0))

            self.update_global(
                r=r,
                global_model=self.model,
                local_models=local_models,
            )

            if r % self.args.test_round == 0:
                # global test loader
                glo_test_acc = self.global_test(
                    model=self.model,
                    loader=self.glo_test_loader,
                )

                # add to log
                self.logs["ROUNDS"].append(r)
                self.logs["LOSSES"].append(train_loss)
                self.logs["GLO_TACCS"].append(glo_test_acc)
                self.logs["LOCAL_TACCS"].extend(per_accs)

                print("[R:{}] [Ls:{}] [TeAc:{}] [PAcBeg:{} PAcAft:{}]".format(
                    r, train_loss, glo_test_acc, per_accs[0], per_accs[-1]
                ))

    def update_local(self, r, model, train_loader, test_loader):
        optimizer = construct_optimizer(
            model, self.args.lr, self.args
        )

        if self.args.local_steps is not None:
            n_total_bs = self.args.local_steps
        elif self.args.local_epochs is not None:
            n_total_bs = max(
                int(self.args.local_epochs * len(train_loader)), 5
            )
        else:
            raise ValueError(
                "local_steps and local_epochs must not be None together"
            )

        model.train()

        loader_iter = iter(train_loader)

        avg_loss = Averager()
        per_accs = []

        for t in range(n_total_bs + 1):
            if t in [0, n_total_bs]:
                per_acc = self.local_test(
                    model=model,
                    loader=test_loader,
                )
                per_accs.append(per_acc)
            if t >= n_total_bs:
                break

            model.train()
            try:
                batch_x, batch_y = loader_iter.next()
            except Exception:
                loader_iter = iter(train_loader)
                batch_x, batch_y = loader_iter.next()

            if self.args.cuda:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

            res = model(batch_x)

            criterion = nn.CrossEntropyLoss()
            if isinstance(res, tuple):
                loss1 = criterion(res[0], batch_y)
                loss2 = criterion(res[1], batch_y)
                loss = loss1 + loss2
            else:
                loss = criterion(res, batch_y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), self.args.max_grad_norm
            )
            optimizer.step()

            avg_loss.add(loss.item())

        loss = avg_loss.item()
        return model, per_accs, loss

    def update_global(self, r, global_model, local_models):
        mean_state_dict = {}

        for name, param in global_model.state_dict().items():
            if "private" in name:
                continue

            vs = []
            for client in local_models.keys():
                vs.append(local_models[client].state_dict()[name])
            vs = torch.stack(vs, dim=0)

            mean_value = vs.mean(dim=0)
            mean_state_dict[name] = mean_value

        global_model.load_state_dict(mean_state_dict, strict=False)

    def local_test(self, model, loader):
        model.eval()

        acc_avg = Averager()

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(loader):
                if self.args.cuda:
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

                res = model(batch_x)

                if isinstance(res, tuple):
                    probs1 = F.softmax(res[0], dim=1)
                    probs2 = F.softmax(res[1], dim=1)
                    probs = probs1 * probs2
                else:
                    probs = F.softmax(res, dim=1)

                acc = count_acc(probs, batch_y)
                acc_avg.add(acc)

        acc = acc_avg.item()
        return acc

    def global_test(self, model, loader):
        model.eval()

        acc_avg = Averager()

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(loader):
                if self.args.cuda:
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

                res = model.global_forward(batch_x)
                if res is not None:
                    probs = F.softmax(res, dim=1)
                    acc = count_acc(probs, batch_y)
                else:
                    acc = 0.0
                acc_avg.add(acc)

        acc = acc_avg.item()
        return acc

    def save_logs(self, fpath):
        all_logs_str = []
        all_logs_str.append(str(self.args))

        logs_str = format_logs(self.logs)
        all_logs_str.extend(logs_str)

        append_to_logs(fpath, all_logs_str)
