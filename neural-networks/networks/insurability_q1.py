"""
Custom FeedForward Neural Network:
- Takes input in the format [38.8, 18.5, 48.] with three input units
- Consists of one layer with two hidden units using a sigmoid activation function
- Employs a custom softmax function for the output layer with three outputs
"""

import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(3, 2)
        self.sigmoid = nn.Sigmoid()
        self.linear_out = nn.Linear(2, 3)

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear_out(x)
        x = self.softmax(x)
        return x

    def softmax(self, x):
        # numerically stable softmax
        exp_vals = torch.exp(x - torch.max(x, dim=1, keepdim=True)[0])
        probabilities = exp_vals / torch.sum(exp_vals, dim=1, keepdim=True)
        return probabilities
