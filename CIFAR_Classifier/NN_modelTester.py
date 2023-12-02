# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import time

import torch
import numpy as np
from torch import nn

import helper_CIFAR10
<<<<<<< HEAD
import helper_GUN_OBJECT

import SimpleNN
# import ssl
=======
import SimpleNN as nn_CIFAR10
import ssl
>>>>>>> 4da08120395bf1ac97d05fd9573c26606a86eab6

ssl._create_default_https_context = ssl._create_unverified_context


def test_model(nn_model: nn.Module, testing_loader, classes_number):

    print(f'Testing model epochs...')

    prediction_matrix = np.zeros((classes_number, classes_number))

    last = time.time()
    with torch.no_grad():
        for load in testing_loader:
            # Input: unloads input data and label
            inputs, labels = load

            # Prediction: generates output
            output = nn_model(inputs)

            # the class with the highest energy is what we choose as prediction
            max_value, predictions = torch.max(output.data, 1)

            prediction_matrix[labels][predictions] += 1

    now = time.time()
    print(f'Finished Testing in {now-last} seconds')

    return prediction_matrix


def main():
    print('NN_modelTester.py\n')
    dataset_choice = int(input('Select a data set\n' +
                               '1. CIFAR10\n' +
                               '2. GUN-OBJECT\n' +
                               '3. _______\n' +
                               'Introduce data set number: '))

    # obtains loaders for training and testing the model
    if dataset_choice == 1:
        _, testing_loader = helper_CIFAR10.get_train_and_test_loader(batch_size=1)
        classes_number = 10
    elif dataset_choice == 2:
        testing_loader = helper_GUN_OBJECT.get_train(batch_size=1)
        classes_number = 4
    else:
        raise Exception('Not Supported')

    path = input('Introduce model PATH: ')

    # creates model and loads data
    nn_model = SimpleNN.SimpleNN()
    nn_model.load_state_dict(torch.load(path))

    # test the model
    prediction_matrix = test_model(nn_model, testing_loader, classes_number)
    print(prediction_matrix)


if __name__ == '__main__':
    main()
