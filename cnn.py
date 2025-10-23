import torch.nn as nn
import torch_base as tb
import torch.nn.functional as nnfun

#Convolutional Neural Net


class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        #define the different layers
        self.conv1=nn.Conv2d(1,32,3,padding=1)
        self.conv2=nn.Conv2d(32,64,3,padding=1)
        self.pool=nn.MaxPool2d(2,2)
        self.fc=nn.Linear(64*7*7,10)

    def forward(self, x):
        #go through the layers
        x=nnfun.relu(self.conv1(x)) #result-28*28*32
        x=self.pool(x) #14*14*32
        x=nnfun.relu(self.conv2(x)) #14*14*64
        x=self.pool(x) #7*7*64
        x=x.view(x.size(0), -1) #flatten
        return self.fc(x)
    
        
    
def run_default(): #Best settings I found
    return tb.run(CNN(), 10, nn.CrossEntropyLoss(), 0.01, 0.9) #recommended settings for highest accuracy I've found

if __name__== "__main__":
    print(f'Convolutional Neural Net Accuracy: {run_default()}%')





