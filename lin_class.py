import torch
import torch.nn as nn
import torch.optim as optim
import torch_base as tb

class LinClass(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc=nn.Linear(784, 10) #define the one-layer neural net

    def forward(self, x):
        x=x.view(x.size(0), -1) #flatten the images
        return self.fc(x) #do a pass through the layer
    
    
def train_linear_classifier(model, train_loader):
    lossfunc=nn.CrossEntropyLoss() #defining the loss function
    optimizer=optim.SGD(model.parameters(), lr=0.1) #the optimizer performs gradient descent with learning rate 0.1
    for images, labels in train_loader:
        optimizer.zero_grad() #reset gradients
        outputs=model(images) #do a prediction
        loss=lossfunc(outputs, labels) #define the loss function on the current values
        loss.backward() #compute gradients of loss through computation graph
        optimizer.step() #perform gradient descent on each tensor


def test_linear_classifier(model, test_loader):
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


def run_linear_classifier():
    model=LinClass()
    train_loader, test_loader=tb.load_data()
    train_linear_classifier(model, train_loader)
    test_linear_classifier(model, test_loader)

run_linear_classifier()