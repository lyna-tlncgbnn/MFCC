import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

# Defining an instance that moves all the tensors(DataLoaders) into the default device
class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def get_dataset(features, labels):
    local_features = []
    local_labels = []
    for i in range(len(labels)):
        local_features.append(features[i])
        local_labels.append(labels[i])
    local_features = torch.stack(local_features)
    local_labels = torch.stack(local_labels)
    local_features = torch.unsqueeze(local_features, dim=1)
    print(local_features.shape)
    return TensorDataset(local_features, local_labels)

def get_sample(features, labels, j):
    local_features = []
    local_labels = []
    for i in range(len(labels)):
        local_features.append(features[j])
        local_labels.append(labels[j])
    local_features = torch.stack(local_features)
    local_labels = torch.stack(local_labels)
    local_features = torch.unsqueeze(local_features, dim=1)
    print(local_features.shape)
    return TensorDataset(local_features, local_labels)

# def get_dataset(features1, features2, features3, labels):
#     local_features1 = []
#     local_features2 = []
#     local_features3 = []
#     local_labels = []
#     for i in range(len(labels)):
#         local_features1.append(features1[i])
#         local_features2.append(features2[i])
#         local_features3.append(features3[i])
#         local_labels.append(labels[i])
#     local_features1 = torch.stack(local_features1)
#     local_features2 = torch.stack(local_features2)
#     local_features3 = torch.stack(local_features3)
#     local_labels = torch.stack(local_labels)
#     local_features1 = torch.unsqueeze(local_features1, dim=1)
#     local_features2 = torch.unsqueeze(local_features2, dim=1)
#     local_features3 = torch.unsqueeze(local_features3, dim=1)
#     return (TensorDataset(local_features1, local_labels), TensorDataset(local_features2, local_labels), TensorDataset(local_features3, local_labels))