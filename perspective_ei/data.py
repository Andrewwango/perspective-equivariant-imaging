import torch
from torchvision.datasets import ImageFolder
import deepinv as dinv
from .spacenet import spacenet_transform, spacenet_loader, spacenet_is_valid_file


def make_dataloaders(dataset_name: str, physics: dinv.physics.Physics, device="cpu") -> tuple:
    """Create torch train and test dataloaders for pansharpening inputs from raw images.
    Currently implemented for loading SpaceNet-4 images. To implement for a different dataset,
    simply add the data in the same folder structure and reimplement the functions in spacenet.py. 

    :param str dataset_name: dataset name e.g. spacenet
    :param dinv.physics.Physics physics: physics operator for simulation
    :param str device: torch device, defaults to "cpu"
    :return tuple: tuple of train_dataloader, test_dataloader
    """
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