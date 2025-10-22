import torch.nn as nn
import torch_base as tb
import torch.nn.functional as nnfun

class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        #define the different layers
        self.conv1=nn.Conv2d(1,32,3,padding=1)
        self.conv2=nn.Conv2d(32,64,3,padding=1)
        self.pool=nn.MaxPool2d(2,2)
        self.fc=nn.Linear(64*7*7,10)

    def forward(self, x):
        x=nnfun.relu(self.conv1(x))
        x=self.pool(x)
        x=nnfun.relu(self.conv2(x))
        x=self.pool(x)
        x=x.view(x.size(0), -1)
        return self.fc(x)
    
    def run_default(self):
        tb.run(self, 10, nn.CrossEntropyLoss(), 0.01, 0.9)
    
    

model=CNN()
model.run_default()





