"""
Data loading and preprocessing module for the plant disease dataset.
"""
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from config.model_config import TRAIN_DIR, VALID_DIR, TEST_DIR, BATCH_SIZE, IMAGE_SHAPE

def create_data_generators(verbose=True):
    """
    Create data generators for training and validation data.
    
    Args:
        verbose (bool): Whether to print information about the generators
        
    Returns:
        tuple: (train_generator, val_generator)
    """
    # Data preprocessing (augmentation only for train)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=10,
        zoom_range=0.1
    )

    valid_datagen = ImageDataGenerator(
        rescale=1./255
    )
    
    # Create train generator
    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SHAPE[:2],
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    # Create validation generator
    val_gen = valid_datagen.flow_from_directory(
        VALID_DIR,
        target_size=IMAGE_SHAPE[:2],
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False  # usually no shuffling in validation
    )
    
    if verbose:
        print("Class indices:")
        print(train_gen.class_indices)
    
    return train_gen, val_gen

def load_test_images(test_dir=TEST_DIR):
    """
    Load test images from the test directory.
    
    Args:
        test_dir (str): Path to the test directory
        
    Returns:
        tuple: (test_images, test_image_paths)
    """
    import numpy as np
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    
    test_images = []
    test_image_paths = []
    
    # Get all image files from the test directory
    for filename in os.listdir(test_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff')):
            img_path = os.path.join(test_dir, filename)
            test_image_paths.append(img_path)
            
            # Load and preprocess the image
            img = load_img(img_path, target_size=IMAGE_SHAPE[:2])
            img_array = img_to_array(img) / 255.0
            test_images.append(img_array)
    
    # Convert to numpy array
    test_images = np.array(test_images)
    
    return test_images, test_image_paths

def get_class_mapping():
    """
    Get the mapping between class indices and class names.
    
    Returns:
        dict: Mapping from class names to indices
    """
    # Create a temporary generator to get class indices
    train_gen, _ = create_data_generators(verbose=False)
    return train_gen.class_indices

# Create generators when imported
train_generator, val_generator = create_data_generators(verbose=False)

# Only print class indices when this file is run directly
if __name__ == "__main__":
    print("Creating data generators...")
    _, _ = create_data_generators(verbose=True)
    print("Data generators created successfully.")
