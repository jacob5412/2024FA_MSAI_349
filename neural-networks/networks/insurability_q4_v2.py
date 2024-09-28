import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()
        self.weights1 = nn.Parameter(torch.randn(3, 2))
        self.weights_out = nn.Parameter(torch.randn(2, 3))

    def forward(self, x):
        x = torch.matmul(x, self.weights1)
        x = self.sigmoid(x)
        x = torch.matmul(x, self.weights_out)
        x = self.softmax(x)
        return x

    def sigmoid(self, x):
        # Numerically stable sigmoid
        return torch.where(
            x >= 0, 1 / (1 + torch.exp(-x)), torch.exp(x) / (1 + torch.exp(x))
        )

    def softmax(self, x):
        # Numerically stable softmax
        exp_vals = torch.exp(x - torch.max(x, dim=1, keepdim=True)[0])
        probabilities = exp_vals / torch.sum(exp_vals, dim=1, keepdim=True)
        return probabilities


class CustomSGD:
    def __init__(self, parameters, lr=0.001):
        self.lr = lr
        self.parameters = list(parameters)

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
