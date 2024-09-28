import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(3, 2, bias=False)
        self.linear_out = nn.Linear(2, 3, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear_out(x)
        x = self.softmax(x)
        return x


class CustomSGD:
    def __init__(self, parameters, lr=0.001):
        self.parameters = list(parameters)
        self.lr = lr

    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

    def step(self):
        with torch.no_grad():
            for param in self.parameters:
                if param.grad is not None:
                    param -= self.lr * param.grad
