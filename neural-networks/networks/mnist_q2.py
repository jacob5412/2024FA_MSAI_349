from torch import nn


class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(784, 32)
        self.relu1 = nn.LeakyReLU()
        self.linear2 = nn.Linear(32, 16)
        self.relu2 = nn.LeakyReLU()
        self.linear_out = nn.Linear(16, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear_out(x)
        x = self.softmax(x)
        return x
