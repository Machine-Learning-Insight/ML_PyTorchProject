# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import torch
import numpy as np

import helper_CIFAR10
import neuralNetwork_CIFAR10 as nn_CIFAR10


if __name__ == '__main__':
    # initializes a neural network
    net = nn_CIFAR10.NeuralNetwork_CIFAR()
    PATH = 'cifar_net.pth'
    net.load_state_dict(torch.load(PATH))
    print(f'Predicting with {PATH}')

    # obtains loaders for training and testing the model
    _, test_loader = helper_CIFAR10.get_train_and_test_loader(batch_size=1)
    classes = helper_CIFAR10.classes

    # a table where [x][y] represents number of x labels predicted a y
    prediction_table = np.zeros((10, 10), dtype=int)

    # number of correct and total predictions
    correct = 0
    total = 0

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            # get the inputs; data is a list of [inputs, labels]
            images, label = data

            # calculate predictions by running images through the network
            prediction_raw = net(images)

            # the class with the highest energy is what we choose as prediction
            _, prediction = torch.max(prediction_raw.data, 1)

            # update stats
            total += 1
            correct += (prediction == label).sum()
            prediction_table[label][prediction] += 1

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    print(prediction_table)