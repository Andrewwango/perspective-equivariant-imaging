import numpy as np
import torch
import deepinv as dinv
from torchvision.datasets.folder import has_file_allowed_extension
from torchvision import transforms
from torchvision.datasets import ImageFolder


# TODO instructions for setting up your own dataset (create folder in data/ with same structure, reimplement functions from spacenet.py)

#spacenet specific code
#TODO docstrings
def spacenet_loader(fn):
    f = np.load(fn)
    return [f["hrms"], f["pan"]]

def spacenet_is_valid_file():
    return lambda f: has_file_allowed_extension(f, (".npz"))

def spacenet_transform():
    transform_hrms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(1024, interpolation=transforms.InterpolationMode.BICUBIC),
    ])
    transform_pan = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(1024, interpolation=transforms.InterpolationMode.BICUBIC),
    ])
    return lambda f: torch.cat([
        transform_hrms(f[0]),
        transform_pan(f[1])
    ], dim=0)

def make_dataloaders(dataset_name: str, physics: dinv.physics.Physics, device="cpu"):

    if dataset_name == "spacenet":
        transform, img_loader, is_valid_file = spacenet_transform(), spacenet_loader, spacenet_is_valid_file()

    dataset_path = dinv.datasets.generate_dataset(
        train_dataset=ImageFolder(f"data/{dataset_name}/subset/train", transform=transform, loader=img_loader, is_valid_file=is_valid_file),
        test_dataset =ImageFolder(f"data/{dataset_name}/subset/test" , transform=transform, loader=img_loader, is_valid_file=is_valid_file),
        physics=physics,
        device=device,
        save_dir=f"data/{dataset_name}",
        num_workers=4 if torch.cuda.is_available() else 0,
        )
        
    train_dataloader = torch.utils.data.DataLoader(
        dinv.datasets.HDF5Dataset(dataset_path, train=True), 
        batch_size=1,
        num_workers=4 if torch.cuda.is_available() else 0,
        shuffle=True
    )
    test_dataloader  = torch.utils.data.DataLoader(
        dinv.datasets.HDF5Dataset(dataset_path, train=False),
        batch_size=1,
        num_workers=4 if torch.cuda.is_available() else 0,
        shuffle=False
    )
    return train_dataloader, test_dataloader