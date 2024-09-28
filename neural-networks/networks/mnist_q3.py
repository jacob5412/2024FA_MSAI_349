from torch import nn


class FeedForwardDropout(nn.Module):
    def __init__(self):
        super(FeedForwardDropout, self).__init__()
        self.linear1 = nn.Linear(784, 32)
        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(32, 16)
        self.relu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(p=0.5)
        self.linear_out = nn.Linear(16, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.linear_out(x)
        x = self.softmax(x)
        return x
