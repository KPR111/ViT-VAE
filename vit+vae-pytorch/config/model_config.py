"""
Configuration parameters for models in PyTorch implementation.
"""

# Data paths (same as TensorFlow version)
TRAIN_DIR = r"C:\Users\Prudvi\Downloads\archive\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train"
VALID_DIR = r"C:\Users\Prudvi\Downloads\archive\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\valid"
TEST_DIR = r"C:\Users\Prudvi\Downloads\archive\test\test"

# Model parameters
IMAGE_SIZE = (224, 224)
IMAGE_CHANNELS = 3
LATENT_DIM = 256
NUM_CLASSES = 38
BATCH_SIZE = 32

# Vision Transformer parameters
PATCH_SIZE = 16  # Divide latent vector into small patches
NUM_PATCHES = LATENT_DIM // PATCH_SIZE
NUM_TRANSFORMER_LAYERS = 6
NUM_HEADS = 8
DROPOUT_RATE = 0.1

# Training parameters
LEARNING_RATE = 1e-4
HYBRID_EPOCHS = 10  # Number of epochs for training the combined model
KL_WEIGHT = 0.01  # Weight factor for KL loss in the combined model
KL_ANNEALING_EPOCHS = 10  # Number of epochs to gradually increase KL weight

# Paths for saving models and checkpoints
VAE_WEIGHTS_PATH = "saved_models/vae"
HYBRID_WEIGHTS_PATH = "saved_models/hybrid"
VAE_CHECKPOINTS_PATH = "checkpoints/vae"
HYBRID_CHECKPOINTS_PATH = "checkpoints/hybrid"

# Normalization parameters
NORMALIZE_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
NORMALIZE_STD = [0.229, 0.224, 0.225]   # ImageNet std

# Device configuration (PyTorch specific)
DEVICE = "cuda"  # Will be updated at runtime based on availability
