import torch.nn as nn
import torch_base as tb

class LinClass(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc=nn.Linear(784, 10) #define the one-layer neural net

    def forward(self, x):
        x=x.view(x.size(0), -1) #flatten the images
        return self.fc(x) #do a pass through the layer
    
    def run_default(self):
        tb.run(self, 10, 0.01)
    
    
model=LinClass()
model.run_default()