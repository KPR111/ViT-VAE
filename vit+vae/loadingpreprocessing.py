# STEP 1: Import Libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# STEP 2: Define Paths
TRAIN_DIR = r"C:\Users\Prudvi\Downloads\archive\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train"
VALID_DIR = r"C:\Users\Prudvi\Downloads\archive\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\valid"

# STEP 3: Data Preprocessing (augmentation only for train)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=10,
    zoom_range=0.1
)

valid_datagen = ImageDataGenerator(
    rescale=1./255
)

# Function to create data generators
def create_data_generators(verbose=True):
    # Create train generator
    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=True
    )

    # Create validation generator
    val_gen = valid_datagen.flow_from_directory(
        VALID_DIR,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False  # usually no shuffling in validation
    )

    if verbose:
        print("Class indices:")
        print(train_gen.class_indices)

    return train_gen, val_gen

# Create generators when imported
train_generator, val_generator = create_data_generators(verbose=False)

# Only print class indices when this file is run directly
if __name__ == "__main__":
    print("Creating data generators...")
    _, _ = create_data_generators(verbose=True)
    print("Data generators created successfully.")
