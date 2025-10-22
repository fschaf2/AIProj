import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_data():
    transform_format=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset=datasets.MNIST(root='./data', train=True, download=True, transform=transform_format)
    test_dataset=datasets.MNIST(root='./data', train=False, download=True, transform=transform_format)

    train_loader=DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader=DataLoader(test_dataset, batch_size=1000, shuffle=False)
    return train_loader, test_loader

def train(model, train_loader, epochs, lr, mom):
    lossfunc=nn.CrossEntropyLoss() #defining the loss function
    optimizer=optim.SGD(model.parameters(), lr=lr, momentum=mom) #the optimizer performs gradient descent with learning rate 0.1
    for i in range(epochs):
        for images, labels in train_loader:
            optimizer.zero_grad() #reset gradients
            outputs=model(images) #do a prediction
            loss=lossfunc(outputs, labels) #define the loss function on the current values
            loss.backward() #compute gradients of loss through computation graph
            optimizer.step() #perform gradient descent on each tensor

def test(model, test_loader):
    correct=0
    total=0
    with torch.no_grad(): #don't keep computation graph for gradients since we don't need them (not training)
        for images, labels in test_loader:
            outputs=model(images)
            _, predicted=outputs.max(1) #find index of predicted value
            total+=labels.size(0) #add size of current batch to total
            correct+=(predicted==labels).sum().item() #add number of correct predictions
    correct_perc=100*(correct/total)
    print(f"Accuracy: {correct_perc}%")

def run(model, epochs, lr, mom):
    train_loader, test_loader=load_data()
    train(model, train_loader, epochs, lr, mom)
    test(model, test_loader)