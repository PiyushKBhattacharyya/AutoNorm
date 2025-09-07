import os
import io
import csv
import math
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def _to_tensor_datasets_from_numpy(X, y, test_size=0.2, random_state=42, scale=True):
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    return train_dataset, test_dataset

def _download_uci_csv(url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        return dest_path
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(resp.content)
    return dest_path

def _load_energy_efficiency(root="./data"):
    # UCI Energy Efficiency Data Set
    # Dataset: 8 features (X1..X8), two targets (Y1: Heating, Y2: Cooling)
    # We will use Y1 (Heating Load) as the regression target.
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
    dest = os.path.join(root, "uci", "energy_efficiency", "ENB2012_data.xlsx")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest):
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(dest, "wb") as f:
            f.write(r.content)
    df = pd.read_excel(dest)
    # Columns might be like: X1..X8, Y1, Y2 (sometimes with trailing spaces)
    df.columns = [str(c).strip() for c in df.columns]
    feature_cols = [c for c in df.columns if c.upper().startswith("X")]
    target_col = "Y1" if "Y1" in df.columns else [c for c in df.columns if c.upper().startswith("Y")][0]
    X = df[feature_cols].values
    y = df[target_col].values
    return _to_tensor_datasets_from_numpy(X, y)

def _load_concrete_strength(root="./data"):
    # UCI Concrete Compressive Strength
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
    dest = os.path.join(root, "uci", "concrete_strength", "Concrete_Data.xls")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest):
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(dest, "wb") as f:
            f.write(r.content)
    df = pd.read_excel(dest)
    # Last column is strength
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return _to_tensor_datasets_from_numpy(X, y)

def _load_airfoil_self_noise(root="./data"):
    # UCI Airfoil Self-Noise
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat"
    dest = os.path.join(root, "uci", "airfoil_self_noise", "airfoil_self_noise.dat")
    _download_uci_csv(url, dest)
    # The file is whitespace separated; last column is the target (Scaled sound pressure level, dB)
    df = pd.read_csv(dest, sep=r"\s+", header=None, engine="python")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return _to_tensor_datasets_from_numpy(X, y)

def get_dataloaders(dataset_name, batch_size=64, custom_transform=None, root="./data"):
    # === Classification datasets ===
    if dataset_name == "MNIST":
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = torchvision.datasets.MNIST(
            root=root, train=True, download=True,
            transform=custom_transform or transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root=root, train=False, download=True,
            transform=custom_transform or transform
        )

    elif dataset_name == "CIFAR10":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = torchvision.datasets.CIFAR10(
            root=root, train=True, download=True,
            transform=custom_transform or train_transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=root, train=False, download=True,
            transform=custom_transform or test_transform
        )

    elif dataset_name == "FashionMNIST":
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = torchvision.datasets.FashionMNIST(
            root=root, train=True, download=True,
            transform=custom_transform or transform
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root=root, train=False, download=True,
            transform=custom_transform or transform
        )

    elif dataset_name == "CIFAR100":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        train_dataset = torchvision.datasets.CIFAR100(
            root=root, train=True, download=True,
            transform=custom_transform or train_transform
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=root, train=False, download=True,
            transform=custom_transform or test_transform
        )

    elif dataset_name == "SVHN":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        ])
        train_dataset = torchvision.datasets.SVHN(
            root=root, split="train", download=True,
            transform=custom_transform or train_transform
        )
        test_dataset = torchvision.datasets.SVHN(
            root=root, split="test", download=True,
            transform=custom_transform or test_transform
        )

    # === Regression datasets ===
    elif dataset_name == "CaliforniaHousing":
        data = fetch_california_housing()
        X, y = data.data.astype(np.float32), data.target.astype(np.float32).reshape(-1, 1)
        train_dataset, test_dataset = _to_tensor_datasets_from_numpy(X, y)

    elif dataset_name == "EnergyEfficiency":
        train_dataset, test_dataset = _load_energy_efficiency(root=root)

    elif dataset_name == "ConcreteStrength":
        train_dataset, test_dataset = _load_concrete_strength(root=root)

    elif dataset_name == "AirfoilSelfNoise":
        train_dataset, test_dataset = _load_airfoil_self_noise(root=root)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader