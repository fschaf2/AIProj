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