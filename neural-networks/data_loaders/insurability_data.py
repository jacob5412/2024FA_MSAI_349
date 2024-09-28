import torch

from .read_data import read_insurability


class CustomInsurabilityDataset(torch.utils.data.Dataset):
    def __init__(self, file, scaler):
        data_set = read_insurability(file)
        transformed_features = scaler.transform(data_set[:, 1:])
        self.features = torch.FloatTensor(transformed_features)
        self.labels = torch.LongTensor(data_set[:, 0])

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
