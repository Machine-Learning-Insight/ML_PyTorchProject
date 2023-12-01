# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import torch
import torch.nn as nn
import torch.optim as optim

import helper_CIFAR10
import neuralNetwork_CIFAR10 as nn_CIFAR10


if __name__ == '__main__':
    # user input
    model_name = input('Introduce name of model: ')
    epoch_number = int(input('Introduce numbers of epoch: '))

    # obtains loaders for training and testing the model
    train_loader, _ = helper_CIFAR10.get_train_and_test_loader(batch_size=4)

    # initializes a neural network
    net = nn_CIFAR10.NeuralNetwork_CIFAR()

    # defines criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # training the neural network
    print(f'Training {model_name} with {epoch_number} epochs...')

    for epoch in range(epoch_number):  # loop over the dataset multiple times
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # obtains prediction
            predictions = net.forward(inputs)

            # calculates loss and gradients
            loss = criterion(predictions, labels)

            # updates parameters
            optimizer.step()
        print(f'- epoch #{epoch+1} completed! Loss: {loss.item()}')

    print(f'Finished Training {model_name}')

    PATH = f'./models/{model_name}.pth'
    torch.save(net.state_dict(), PATH)
    print(f'{model_name} was saved at {PATH}')
