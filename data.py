import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_dataloaders(dataset_name, batch_size=64, custom_transform=None):
    if dataset_name == "MNIST":
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        train_dataset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True,
            transform=custom_transform or transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True,
            transform=custom_transform or transform
        )

    elif dataset_name == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        train_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True,
            transform=custom_transform or transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True,
            transform=custom_transform or transform
        )

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader
