import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# obtains the classes ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck');


def get_train_and_test_loader(batch_size: int) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:

    # defines a transform that converts PIL images with three channels from range [0, 1] to [-1, 1]
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # optains the train and test set
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    return (torch.utils.data.DataLoader(train_set, batch_size, shuffle=True),
            torch.utils.data.DataLoader(test_set, batch_size, shuffle=False))


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()
