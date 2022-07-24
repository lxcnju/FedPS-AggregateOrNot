import torch
import torch.nn as nn

from utils import weights_init


# reference: http://www.lamda.nju.edu.cn/lixc/papers/FedPS-CoRR-Lixc.pdf
# Figure.4


def get_basic_net(net, n_classes):
    if net == "MLPNet":
        model = MLPNet(n_classes)
    elif net == "LeNet":
        model = LeNet(n_classes)
    elif net == "TFCNN":
        model = TFCNN(n_classes)
    elif net == "VGG11":
        model = VGG11(n_classes)
    elif net == "FeMnistNet":
        model = FeMnistNet(n_classes)
    elif net == "CharLSTM":
        model = CharLSTM(n_classes)
    else:
        raise ValueError("No such net: {}".format(net))

    model.apply(weights_init)
    return model


class Reshape(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, xs):
        return xs.reshape((xs.shape[0], -1))


class MLPNet(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.n_classes = n_classes

        self.encoder = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 128),
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, n_classes)
        )

    def forward(self, xs):
        xs = xs.reshape((-1, 784))
        code = self.encoder(xs)
        logits = self.classifier(code)
        return code, logits


class LeNet(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.n_classes = n_classes

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            Reshape(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 120),
            nn.ReLU(True),
            nn.Linear(120, 84),
            nn.ReLU(True),
            nn.Linear(84, n_classes)
        )

    def forward(self, xs):
        code = self.encoder(xs)
        logits = self.classifier(code)
        return code, logits


class FeMnistNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            Reshape(),
        )

        # 2048 --> 512
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(True),
            nn.Linear(512, n_classes)
        )

    def forward(self, xs):
        code = self.encoder(xs)
        logits = self.classifier(code)
        return code, logits


class TFCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            Reshape(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(True),
            nn.Linear(128, n_classes)
        )

    def forward(self, xs):
        code = self.encoder(xs)
        logits = self.classifier(code)
        return code, logits


class VGG11(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.n_classes = n_classes

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Reshape(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, n_classes),
        )

    def forward(self, xs):
        code = self.encoder(xs)
        logits = self.classifier(code)
        return code, logits


class InnerLSTM(nn.Module):
    def __init__(self, lstm):
        super().__init__()
        self.lstm = lstm
        self.lstm_size = lstm.hidden_size

    def forward(self, xs):
        bs = xs.shape[0]
        h0 = torch.zeros(2, bs, self.lstm_size)
        c0 = torch.zeros(2, bs, self.lstm_size)

        h0 = h0.to(device=xs.device)
        c0 = c0.to(device=xs.device)

        outputs, _ = self.lstm(xs, (h0, c0))
        output = outputs[:, -1, :]
        return output


class CharLSTM(nn.Module):
    """ StackedLSTM for NLP
    """

    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

        self.n_vocab = 81
        self.w_dim = 8
        self.lstm_size = 256

        embeddings = nn.Embedding(self.n_vocab, self.w_dim)
        lstm = nn.LSTM(
            input_size=self.w_dim,
            hidden_size=self.lstm_size,
            num_layers=2,
            batch_first=True,
        )
        inner_lstm = InnerLSTM(lstm)

        self.encoder = nn.Sequential(
            embeddings,
            inner_lstm,
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.lstm_size, self.n_classes),
        )

    def forward(self, xs):
        code = self.encoder(xs)
        logits = self.classifier(code)
        return code, logits


if __name__ == "__main__":
    xs = torch.randn(32, 1, 28, 28)
    net = LeNet(n_classes=10)
    code, logits = net(xs)
    print(code.shape, logits.shape)

    xs = torch.randn(32, 1, 28, 28)
    net = FeMnistNet(n_classes=10)
    code, logits = net(xs)
    print(code.shape, logits.shape)
