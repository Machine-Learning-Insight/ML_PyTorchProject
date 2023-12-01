# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import torch
import torchvision

import helper_CIFAR10
import neuralNetwork_CIFAR10 as nn_CIFAR10

# initializes a neural network
net = nn_CIFAR10.NeuralNetwork_CIFAR()
net.load_state_dict(torch.load('cifar_net.pth'))

if __name__ == '__main__':
    # obtains loaders for training and testing the model
    _, test_loader = helper_CIFAR10.get_train_and_test_loader(batch_size=4)
    classes = helper_CIFAR10.classes

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')