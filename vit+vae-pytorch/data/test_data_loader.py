"""
Data loading module for test images with class names in filenames.
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import re
from config.model_config import TEST_DIR, BATCH_SIZE, IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD

class FlatTestDataset(Dataset):
    """
    Dataset class for test images in a flat directory structure
    where class names are part of the filenames.
    """
    def __init__(self, root_dir, transform=None, class_mapping=None):
        """
        Initialize the dataset.

        Args:
            root_dir (str): Directory with all the test images
            transform (callable, optional): Optional transform to be applied on images
            class_mapping (dict, optional): Mapping from class names to indices
        """
        self.root_dir = root_dir
        self.transform = transform

        # Get all image files from the test directory
        self.samples = []
        self.class_names = set()

        print(f"Scanning directory: {root_dir}")
        files = os.listdir(root_dir)
        print(f"Found {len(files)} files in directory")

        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff')):
                img_path = os.path.join(root_dir, filename)

                # Extract class name from filename (assuming format like "AppleScab3.JPG")
                # First, remove the file extension
                base_name = os.path.splitext(filename)[0]

                # Extract the alphabetic part (class name) from the filename
                # This will handle formats like "AppleScab3" -> "AppleScab"
                class_name = ''.join([c for c in base_name if c.isalpha()])

                print(f"File: {filename}, Extracted class: {class_name}")

                if class_name:  # Only add if we extracted a valid class name
                    self.class_names.add(class_name)
                    self.samples.append((img_path, class_name))

        # Create class to index mapping if not provided
        if class_mapping is None:
            self.class_names = sorted(list(self.class_names))
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}
        else:
            self.class_to_idx = class_mapping

        # Print class mapping for debugging
        print("Class mapping from training dataset:")
        for cls_name, idx in self.class_to_idx.items():
            print(f"  {idx}: {cls_name}")

        # Update samples with class indices
        # Try to match class names more flexibly
        self.samples_with_indices = []
        for path, cls_name in self.samples:
            # Try exact match first
            if cls_name in self.class_to_idx:
                self.samples_with_indices.append((path, self.class_to_idx[cls_name]))
            else:
                # Try case-insensitive match
                matched = False
                for train_cls_name in self.class_to_idx.keys():
                    if cls_name.lower() == train_cls_name.lower():
                        print(f"Case-insensitive match: {cls_name} -> {train_cls_name}")
                        self.samples_with_indices.append((path, self.class_to_idx[train_cls_name]))
                        matched = True
                        break

                # Try partial match (if test class name is contained in training class name)
                if not matched:
                    for train_cls_name in self.class_to_idx.keys():
                        # Convert test class name to match training format
                        # Example: "AppleScab" -> look for "Apple" and "scab" in "Apple___Apple_scab"

                        # Handle special cases
                        if cls_name == "AppleCedarRust":
                            if "Apple___Cedar_apple_rust" in train_cls_name:
                                print(f"Special match: {cls_name} -> {train_cls_name}")
                                self.samples_with_indices.append((path, self.class_to_idx[train_cls_name]))
                                matched = True
                                break
                        elif cls_name == "AppleScab":
                            if "Apple___Apple_scab" in train_cls_name:
                                print(f"Special match: {cls_name} -> {train_cls_name}")
                                self.samples_with_indices.append((path, self.class_to_idx[train_cls_name]))
                                matched = True
                                break
                        elif cls_name == "CornCommonRust":
                            if "Corn_(maize)___Common_rust_" in train_cls_name:
                                print(f"Special match: {cls_name} -> {train_cls_name}")
                                self.samples_with_indices.append((path, self.class_to_idx[train_cls_name]))
                                matched = True
                                break
                        elif cls_name == "PotatoEarlyBlight":
                            if "Potato___Early_blight" in train_cls_name:
                                print(f"Special match: {cls_name} -> {train_cls_name}")
                                self.samples_with_indices.append((path, self.class_to_idx[train_cls_name]))
                                matched = True
                                break
                        elif cls_name == "PotatoHealthy":
                            if "Potato___healthy" in train_cls_name:
                                print(f"Special match: {cls_name} -> {train_cls_name}")
                                self.samples_with_indices.append((path, self.class_to_idx[train_cls_name]))
                                matched = True
                                break
                        elif cls_name == "TomatoEarlyBlight":
                            if "Tomato___Early_blight" in train_cls_name:
                                print(f"Special match: {cls_name} -> {train_cls_name}")
                                self.samples_with_indices.append((path, self.class_to_idx[train_cls_name]))
                                matched = True
                                break
                        elif cls_name == "TomatoHealthy":
                            if "Tomato___healthy" in train_cls_name:
                                print(f"Special match: {cls_name} -> {train_cls_name}")
                                self.samples_with_indices.append((path, self.class_to_idx[train_cls_name]))
                                matched = True
                                break
                        elif cls_name == "TomatoYellowCurlVirus":
                            if "Tomato___Tomato_Yellow_Leaf_Curl_Virus" in train_cls_name:
                                print(f"Special match: {cls_name} -> {train_cls_name}")
                                self.samples_with_indices.append((path, self.class_to_idx[train_cls_name]))
                                matched = True
                                break

                        # General partial matching as a fallback
                        elif cls_name.lower() in train_cls_name.lower() or train_cls_name.lower() in cls_name.lower():
                            print(f"Partial match: {cls_name} -> {train_cls_name}")
                            self.samples_with_indices.append((path, self.class_to_idx[train_cls_name]))
                            matched = True
                            break

                if not matched:
                    print(f"No match found for class: {cls_name}")

        # Update samples
        self.samples = self.samples_with_indices

        print(f"Final dataset size: {len(self.samples)} images")

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

        return img, target

def create_flat_test_loader(test_dir=TEST_DIR, batch_size=BATCH_SIZE, class_mapping=None):
    """
    Create a data loader for test images in a flat directory structure.

    Args:
        test_dir (str): Path to the test directory
        batch_size (int): Batch size for the data loader
        class_mapping (dict, optional): Mapping from class names to indices

    Returns:
        tuple: (test_loader, class_to_idx)
    """
    # Define transform for test data
    test_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),  # Ensure exact size
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])

    # Create test dataset
    test_dataset = FlatTestDataset(
        root_dir=test_dir,
        transform=test_transform,
        class_mapping=class_mapping
    )

    # Create test data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return test_loader, test_dataset.class_to_idx

def check_test_directory(test_dir=TEST_DIR):
    """
    Check the test directory and print information about its contents.

    Args:
        test_dir (str): Path to the test directory
    """
    print(f"Checking test directory: {test_dir}")

    # Check if directory exists
    if not os.path.exists(test_dir):
        print(f"Directory does not exist: {test_dir}")
        return

    # List all files
    files = os.listdir(test_dir)
    print(f"Found {len(files)} files in directory")

    # Count image files
    image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff'))]
    print(f"Found {len(image_files)} image files")

    # Print first 10 image files
    if image_files:
        print("Sample image files:")
        for i, f in enumerate(image_files[:10]):
            print(f"  {i+1}. {f}")

        # Try to extract class names
        print("\nExtracted class names:")
        for i, f in enumerate(image_files[:10]):
            base_name = os.path.splitext(f)[0]
            class_name = ''.join([c for c in base_name if c.isalpha()])
            print(f"  {i+1}. {f} -> {class_name}")
    else:
        print("No image files found in directory")

# For testing
if __name__ == "__main__":
    # First check the test directory
    check_test_directory()

    # Then try to create the test loader
    test_loader, class_to_idx = create_flat_test_loader()
    print(f"\nTest dataset size: {len(test_loader.dataset)} images")
    print(f"Number of classes: {len(class_to_idx)}")
    print("Class mapping:")
    for cls_name, idx in class_to_idx.items():
        print(f"  {idx}: {cls_name}")
