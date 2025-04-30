"""
Data loading and preprocessing module for the plant disease dataset using PyTorch.
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from config.model_config import TRAIN_DIR, VALID_DIR, TEST_DIR, BATCH_SIZE, IMAGE_SIZE

class PlantDiseaseDataset(Dataset):
    """
    Dataset class for plant disease images.
    """
    def __init__(self, root_dir, transform=None, target_transform=None):
        """
        Initialize the dataset.

        Args:
            root_dir (str): Directory with all the images organized in class folders
            transform (callable, optional): Optional transform to be applied on images
            target_transform (callable, optional): Optional transform to be applied on labels
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform

        # Get all class folders
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Get all image paths and corresponding labels
        self.samples = []
        for target_class in self.classes:
            class_dir = os.path.join(root_dir, target_class)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff')):
                    path = os.path.join(class_dir, fname)
                    self.samples.append((path, self.class_to_idx[target_class]))

    def __len__(self):
        """Return the total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample

        Returns:
            tuple: (image, label) where label is the class index
        """
        path, target = self.samples[idx]

        # Load image
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        # Apply target transform
        if self.target_transform:
            target = self.target_transform(target)

        return img, target

def create_data_loaders(verbose=True):
    """
    Create data loaders for training and validation data.

    Args:
        verbose (bool): Whether to print information about the loaders

    Returns:
        tuple: (train_loader, val_loader, class_to_idx)
    """
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),  # Ensure exact size
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),  # Ensure exact size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = PlantDiseaseDataset(
        root_dir=TRAIN_DIR,
        transform=train_transform
    )

    val_dataset = PlantDiseaseDataset(
        root_dir=VALID_DIR,
        transform=val_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    if verbose:
        print(f"Training dataset size: {len(train_dataset)} images")
        print(f"Validation dataset size: {len(val_dataset)} images")
        print(f"Number of classes: {len(train_dataset.classes)}")
        print("Class mapping:")
        for cls_name, idx in train_dataset.class_to_idx.items():
            print(f"  {idx}: {cls_name}")

    return train_loader, val_loader, train_dataset.class_to_idx

def load_test_images(test_dir=TEST_DIR):
    """
    Load test images from the test directory.

    Args:
        test_dir (str): Path to the test directory

    Returns:
        tuple: (test_images, test_image_paths)
    """
    # Define transform
    test_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),  # Ensure exact size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_images = []
    test_image_paths = []

    # Get all image files from the test directory
    for filename in os.listdir(test_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff')):
            img_path = os.path.join(test_dir, filename)
            test_image_paths.append(img_path)

            # Load and preprocess the image
            with open(img_path, 'rb') as f:
                img = Image.open(f).convert('RGB')
                img_tensor = test_transform(img)
                test_images.append(img_tensor)

    # Stack tensors
    if test_images:
        test_images = torch.stack(test_images)
    else:
        test_images = torch.tensor([])

    return test_images, test_image_paths

def get_class_mapping():
    """
    Get the mapping between class indices and class names.

    Returns:
        dict: Mapping from class names to indices
    """
    # Create a temporary dataset to get class indices
    dataset = PlantDiseaseDataset(root_dir=TRAIN_DIR)
    return dataset.class_to_idx

# Create global variables for easy access
train_loader, val_loader, class_to_idx = None, None, None

def initialize_data_loaders(verbose=False):
    """Initialize the global data loaders."""
    global train_loader, val_loader, class_to_idx
    train_loader, val_loader, class_to_idx = create_data_loaders(verbose=verbose)
    return train_loader, val_loader, class_to_idx

# Only print class indices when this file is run directly
if __name__ == "__main__":
    print("Creating data loaders...")
    train_loader, val_loader, class_to_idx = create_data_loaders(verbose=True)
    print("Data loaders created successfully.")
