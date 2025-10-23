import torch.nn as nn
import torch_base as tb

#Multi-Layer Perceptron



class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential( #define the layers, mapping over to 10 through all of them
            nn.Linear(784,256),
            nn.ReLU(), #introduce nonlinearity
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )

    def forward(self, x):
        x=x.view(x.size(0), -1) #flatten the images
        return self.layers(x) #do a pass through the layer
    
    
def run_default(): #Best settings I found
    return tb.run(MLP(), 10, nn.CrossEntropyLoss(), 0.01, 0)

if __name__== "__main__":
    print(f'Multilayer Perceptron Accuracy: {run_default()}%')





