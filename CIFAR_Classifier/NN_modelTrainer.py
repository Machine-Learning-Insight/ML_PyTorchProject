# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import torch
import torch.nn as nn
import torch.optim as optim
import helper_CIFAR10
import helper_GUN_OBJECT
import SimpleNN
import time
from pathlib import Path
import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


device_cuda = torch.device('cuda:0') if (torch.cuda.is_available()) else None


def train_model(nn_model: nn.Module, training_loader, epoch_number):

    print(f'Training model with {epoch_number} epochs...')
    last = time.time()

    # defines optimizer
    optimizer = optim.SGD(nn_model.parameters(), lr=0.001, momentum=0.9)

    # defines criterion
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epoch_number):  # trains multiple epochs

        for load in training_loader:
            # Cleanup: sets the model's parameters' gradients to zero
            optimizer.zero_grad()

            # Input: unloads input data and label
            if device_cuda:
                inputs, labels = load[0].to(device_cuda), load[1].to(device_cuda)
            else:
                inputs, labels = load

            # Prediction: generates output
            output = nn_model(inputs)

            # Optimization: calculates loss and gradients to optimize model
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

        now = time.time()
        print(f'- epoch #{epoch + 1} completed!\tTime: {(now - last):.2f}')
        last = time.time()

    print(f'Finished Training')


def main():
    print('NN_modelTrainer.py\n')
    dataset_choice = int(input('Select a data set\n' +
                               '1. CIFAR10\n' +
                               '2. GUN-OBJECT\n' +
                               '3. _______\n' +
                               'Introduce data set number: '))

    # obtains loaders for training and testing the model
    if dataset_choice == 1:
        training_loader, _ = helper_CIFAR10.get_train_and_test_loader(batch_size=4)
    elif dataset_choice == 2:
        training_loader = helper_GUN_OBJECT.get_train(batch_size=4)
    else:
        raise Exception('Not Supported')

    model_name = input('Introduce model name: ')
    epoch_number = int(input('Introduce numbers of epoch: '))

    # creates model and trains it
    nn_model = SimpleNN.SimpleNN()
    if device_cuda:
        nn_model.to(device_cuda)

    train_model(nn_model, training_loader, epoch_number)

    # saves model
    if not os.path.exists("models"):
        os.mkdir('models')

    relative_path = f'./models/{model_name}.pth'
    torch.save(nn_model.state_dict(), relative_path)
    print(f'{model_name} was saved at {Path(relative_path).resolve()}')


if __name__ == '__main__':
    main()
