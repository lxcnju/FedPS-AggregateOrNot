import torch
import torch.nn as nn
import torch.nn.functional as F

from basic_nets import Reshape
from basic_nets import get_basic_net

from utils import weights_init

# reference: http://www.lamda.nju.edu.cn/lixc/papers/FedPS-CoRR-Lixc.pdf
# Figure.1 & Figure.2 & Figure.4


def get_ps_net(ps_type, net, split_layer, n_classes):
    """
    ps_type: ["SPNet", "PSNet", "SP2Net", "PS2Net"]
    net: ["MLPNet", "LeNet", "TFCNN", "VGG11",
          "SACNN", "SACNNSmall", "CharLSTM", "FeMnistNet"]
         ["VGG11FinePS", "CharLSTMFinePS"]
    split_layer: int, 0-1, 0-5, 0-2
    n_classes: int
    """
    if ps_type == "SPNet":
        ps_net = SPNet(net, split_layer, n_classes)
    elif ps_type == "PSNet":
        ps_net = PSNet(net, split_layer, n_classes)
    elif ps_type == "SP2Net":
        ps_net = SP2Net(net, split_layer, n_classes)
    elif ps_type == "PS2Net":
        ps_net = PS2Net(net, split_layer, n_classes)
    else:
        raise ValueError("No such ps_type: {}".format(ps_type))

    ps_net.apply(weights_init)
    return ps_net


class InnerLSTM1(nn.Module):
    def __init__(self, lstm):
        super().__init__()
        self.lstm = lstm
        self.lstm_size = lstm.hidden_size

    def forward(self, xs):
        bs = xs.shape[0]
        h0 = torch.zeros(1, bs, self.lstm_size)
        c0 = torch.zeros(1, bs, self.lstm_size)

        h0 = h0.to(device=xs.device)
        c0 = c0.to(device=xs.device)

        outputs, _ = self.lstm(xs, (h0, c0))
        return outputs


class InnerLSTM2(nn.Module):
    def __init__(self, lstm):
        super().__init__()
        self.lstm = lstm
        self.lstm_size = lstm.hidden_size

    def forward(self, xs):
        bs = xs.shape[0]
        h0 = torch.zeros(1, bs, self.lstm_size)
        c0 = torch.zeros(1, bs, self.lstm_size)

        h0 = h0.to(device=xs.device)
        c0 = c0.to(device=xs.device)

        outputs, _ = self.lstm(xs, (h0, c0))
        output = outputs[:, -1, :]
        return output


class LayeredCharLSTM(nn.Module):
    """ StackedLSTM for NLP
    # Figure.4
    """

    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

        self.n_vocab = 81
        self.w_dim = 8
        self.lstm_size = 256

        self.names = [
            "embedding", "lstm1", "lstm2", "classifier"
        ]

        self.layers = nn.ModuleList()

        embeddings = nn.Embedding(self.n_vocab, self.w_dim)
        self.layers.append(embeddings)

        lstm1 = nn.LSTM(
            input_size=self.w_dim,
            hidden_size=self.lstm_size,
            num_layers=1,
            batch_first=True,
        )
        inner_lstm1 = InnerLSTM1(lstm1)
        self.layers.append(inner_lstm1)

        lstm2 = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=1,
            batch_first=True,
        )
        inner_lstm2 = InnerLSTM2(lstm2)
        self.layers.append(inner_lstm2)

        classifier = nn.Sequential(
            nn.Linear(self.lstm_size, self.n_classes),
        )
        self.layers.append(classifier)

    def forward(self, xs):
        for i in range(len(self.names)):
            xs = self.layers[i](xs)
        return xs


class LayeredVGG11(nn.Module):
    # Figure.4
    def __init__(self, n_classes=10):
        super().__init__()
        self.n_classes = n_classes

        self.names = [
            "encoder1", "encoder2", "encoder3", "encoder4",
            "classifier1", "classifier2"
        ]

        self.layers = nn.ModuleList()

        encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layers.append(encoder1)

        encoder2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layers.append(encoder2)

        encoder3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layers.append(encoder3)

        encoder4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Reshape(),
        )
        self.layers.append(encoder4)

        classifier1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
        )
        self.layers.append(classifier1)

        classifier2 = nn.Sequential(
            nn.Linear(256, n_classes),
        )
        self.layers.append(classifier2)

    def forward(self, xs):
        for i in range(len(self.names)):
            xs = self.layers[i](xs)
        return xs


class TwoLayeredNet(nn.Module):
    def __init__(self, net, n_classes=10):
        super().__init__()
        self.n_classes = n_classes
        model = get_basic_net(net, n_classes)

        self.names = ["encoder", "classifier"]
        self.layers = nn.ModuleList()
        self.layers.append(model.encoder)
        self.layers.append(model.classifier)

    def forward(self, xs):
        for i in range(len(self.names)):
            xs = self.layers[i](xs)
        return xs


def get_layered_net(net, n_classes):
    if net == "VGG11FinePS":
        model = LayeredVGG11(n_classes)
    elif net == "CharLSTMFinePS":
        model = LayeredCharLSTM(n_classes)
    else:
        model = TwoLayeredNet(net, n_classes)

    model.apply(weights_init)
    return model


class SPNet(nn.Module):
    """
    SP
    Figure.1
    share --> private
    split_layer: split to shared / private
    """

    def __init__(
        self, net, split_layer, n_classes=10
    ):
        super().__init__()
        self.split_layer = split_layer

        model = get_layered_net(net, n_classes)

        assert 0 <= split_layer < len(model.names)

        self.shared_layers = nn.ModuleList()
        self.private_layers = nn.ModuleList()

        for i in range(len(model.names)):
            if i < split_layer:
                self.shared_layers.append(model.layers[i])
            else:
                self.private_layers.append(model.layers[i])

    def forward(self, xs):
        for layer in self.shared_layers:
            xs = layer(xs)

        for layer in self.private_layers:
            xs = layer(xs)

        return xs

    def global_forward(self, xs):
        if len(self.private_layers) == 0:
            return self.forward(xs)
        else:
            return None


class PSNet(nn.Module):
    """
    PS
    Figure.1
    private --> shared
    split_layer: split to shared / private
    """

    def __init__(
        self, net, split_layer, n_classes=10
    ):
        super().__init__()
        self.split_layer = split_layer

        model = get_layered_net(net, n_classes)

        assert 0 <= split_layer < len(model.names)

        self.private_layers = nn.ModuleList()
        self.shared_layers = nn.ModuleList()

        for i in range(len(model.names)):
            if i < split_layer:
                self.private_layers.append(model.layers[i])
            else:
                self.shared_layers.append(model.layers[i])

    def forward(self, xs):
        for layer in self.private_layers:
            xs = layer(xs)

        for layer in self.shared_layers:
            xs = layer(xs)

        return xs

    def global_forward(self, xs):
        if len(self.private_layers) == 0:
            return self.forward(xs)
        else:
            return None


class SP2Net(nn.Module):
    """
    SSP
    Figure.1
    shared --> {private, shared}
    split_layer: split to shared / private
    """

    def __init__(
        self, net, split_layer, n_classes=10
    ):
        super().__init__()
        self.split_layer = split_layer

        model = get_layered_net(net, n_classes)
        pmodel = get_layered_net(net, n_classes)

        assert 0 <= split_layer < len(model.names)

        self.shared_layers = nn.ModuleList()
        self.private_layers = nn.ModuleList()

        for i in range(len(model.names)):
            self.shared_layers.append(model.layers[i])
            if i >= split_layer:
                self.private_layers.append(pmodel.layers[i])

    def forward(self, xs):
        for i in range(self.split_layer):
            xs = self.shared_layers[i](xs)

        private_xs = xs
        for i in range(self.split_layer, len(self.shared_layers)):
            xs = self.shared_layers[i](xs)

        for i in range(len(self.private_layers)):
            private_xs = self.private_layers[i](private_xs)

        return xs, private_xs

    def global_forward(self, xs):
        for layer in self.shared_layers:
            xs = layer(xs)
        return xs


class PS2Net(nn.Module):
    """
    SPS
    Figure.1
    {shared, private} --> shared
    split_layer: split to shared / private
    """

    def __init__(
        self, net, split_layer, n_classes=10
    ):
        super().__init__()
        self.split_layer = split_layer

        model = get_layered_net(net, n_classes)
        pmodel = get_layered_net(net, n_classes)

        assert 0 <= split_layer < len(model.names)

        self.shared_layers = nn.ModuleList()
        self.private_layers = nn.ModuleList()

        for i in range(len(model.names)):
            self.shared_layers.append(model.layers[i])
            if i < split_layer:
                self.private_layers.append(pmodel.layers[i])

    def forward(self, xs):
        private_xs = xs

        for i in range(self.split_layer):
            xs = self.shared_layers[i](xs)

        for i in range(self.split_layer):
            private_xs = self.private_layers[i](private_xs)

        xs = (0.5 * (private_xs + xs)).type(xs.dtype)

        for i in range(self.split_layer, len(self.shared_layers)):
            xs = self.shared_layers[i](xs)

        return xs

    def global_forward(self, xs):
        for layer in self.shared_layers:
            xs = layer(xs)
        return xs


class AutoPSNet(nn.Module):
    """
    Figure.2
    shared, private --> cross-stitch
    """

    def __init__(
        self, net, way, n_classes=10
    ):
        super().__init__()
        self.way = way

        model = get_layered_net(net, n_classes)
        pmodel = get_layered_net(net, n_classes)

        self.shared_layers = nn.ModuleList()
        self.private_layers = nn.ModuleList()

        self.shared_coefs = nn.ParameterList()

        for i in range(len(model.names)):
            self.shared_layers.append(model.layers[i])
            self.private_layers.append(pmodel.layers[i])

        if "cs" in self.way:
            for i in range(len(model.names)):
                self.shared_coefs.append(
                    nn.Parameter(0.5 * torch.ones(2, 2))
                )
        elif "attn" in self.way:
            for i in range(len(model.names)):
                self.shared_coefs.append(
                    nn.Parameter(0.5 * torch.ones(1, 2))
                )
        elif "gs" in self.way:
            for i in range(len(model.names)):
                self.shared_coefs.append(
                    nn.Parameter(0.5 * torch.ones(1, 2))
                )

    def forward(self, xs):
        n_layer = len(self.shared_layers)

        sxs = xs
        pxs = xs
        for i in range(n_layer):
            sxs = self.shared_layers[i](sxs)
            pxs = self.private_layers[i](pxs)

            if "cs" in self.way:
                coefs = self.shared_coefs[i].softmax(dim=-1)

                x1 = coefs[0][0] * sxs + coefs[0][1] * pxs
                x2 = coefs[1][0] * sxs + coefs[1][1] * pxs
                sxs = x1
                pxs = x2
            elif "attn" in self.way:
                coefs = self.shared_coefs[i].softmax(dim=-1)

                x1 = coefs[0][0] * sxs + coefs[0][1] * pxs
                sxs = x1
                pxs = x1
            elif "gs" in self.way:
                coefs = self.shared_coefs[i]

                code = F.gumbel_softmax(
                    coefs, tau=1.0, hard=True, eps=1e-10
                )

                x1 = code[0][0] * sxs + code[0][1] * pxs
                sxs = x1
                pxs = x1

        return sxs, pxs

    def global_forward(self, xs):
        n_layer = len(self.shared_layers)

        for i in range(n_layer):
            xs = self.shared_layers[i](xs)

        return xs


if __name__ == "__main__":
    model = LayeredCharLSTM(n_classes=10)

    xs = torch.randint(0, 80, (32, 40))
    logits = model(xs)
    print(logits.shape)
