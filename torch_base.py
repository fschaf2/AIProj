import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as nnfun
import numpy as np
import np_base as base
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

def train(model, train_loader, epochs, lossfunc, lr, mom):
    optimizer=optim.SGD(model.parameters(), lr=lr, momentum=mom) #the optimizer performs gradient descent with learning rate 0.1
    for i in range(epochs):
        #print(f"epoch {i}")
        for images, labels in train_loader:
            optimizer.zero_grad() #reset gradients
            outputs=model(images) #do a prediction
            #define the loss function on the current values. Use one-hot encoding if MSE since that's the format it needs.
            if isinstance(lossfunc, nn.MSELoss):
                loss=lossfunc(outputs, nnfun.one_hot(labels, num_classes=outputs.size(1)).float())
            else:
                loss=lossfunc(outputs, labels) 
            loss.backward() #compute gradients of loss through computation graph
            optimizer.step() #perform gradient descent on each tensor

def test(model, test_loader):
    print("testing now")
    correct=0
    total=0
    true_list=[]
    pred_list=[]
    with torch.no_grad(): #don't keep computation graph for gradients since we don't need them (not training)
        for images, labels in test_loader:
            outputs=model(images)
            _, predicted=outputs.max(1) #find index of predicted value
            true_list.append(labels.cpu())
            pred_list.append(predicted.cpu())
    true_np=torch.cat(true_list).numpy()
    pred_np=torch.cat(pred_list).numpy()
    base.plot_cm(true_np, pred_np, type(model).__name__)
    correct_perc=(np.mean(pred_np==true_np))*100
    return correct_perc

def run(model, epochs, lossfunc, lr, mom):
    train_loader, test_loader=load_data()
    train(model, train_loader, epochs, lossfunc, lr, mom)
    return test(model, test_loader)