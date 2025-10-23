import torch.nn as nn
import torch_base as tb

class LinClass(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc=nn.Linear(784, 10) #define the one-layer neural net

    def forward(self, x):
        x=x.view(x.size(0), -1) #flatten the images
        return self.fc(x) #do a pass through the layer
    


def run_default():
    return tb.run(LinClass(), 10, nn.MSELoss(), 0.01, 0)

if __name__== "__main__":
    print(f'Linear Classifier Accuracy: {run_default()}%')