import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class CustomDataSet(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        image = Image.open(self.file_paths[index]).convert('RGB')
        label = torch.tensor(self.labels[index])

        if self.transform:
            image = self.transform(image)

        return image, label


file_paths = []
labels = []

# add grenades
for filename in os.listdir("data\\gun-object\\grenade"):
    file_paths.append(f"data\\gun-object\\grenade\\{filename}")
    labels.append(0)

# add knife
for filename in os.listdir("data\\gun-object\\knife"):
    file_paths.append(f"data\\gun-object\\knife\\{filename}")
    labels.append(1)

# add pistol
for filename in os.listdir("data\\gun-object\\pistol"):
    file_paths.append(f"data\\gun-object\\pistol\\{filename}")
    labels.append(2)

# add riffle
for filename in os.listdir("data\\gun-object\\rifle"):
    file_paths.append(f"data\\gun-object\\rifle\\{filename}")
    labels.append(3)


def get_train(batch_size: int) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    # defines a transform that converts PIL images with three channels from range [0, 255] to [-1, 1]
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((255 / 2, 255 / 2, 255 / 2), (255 / 2, 255 / 2, 255 / 2))])

    custom_dataset = CustomDataSet(file_paths, labels, transform=transform)

    return torch.utils.data.DataLoader(custom_dataset, batch_size, shuffle=True)