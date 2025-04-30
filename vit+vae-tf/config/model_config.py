# Configuration parameters for models

# Data paths
TRAIN_DIR = r"C:\Users\Prudvi\Downloads\archive\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train"
VALID_DIR = r"C:\Users\Prudvi\Downloads\archive\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\valid"
TEST_DIR = r"C:\Users\Prudvi\Downloads\archive\test\test"

# Model parameters
IMAGE_SHAPE = (224, 224, 3)
LATENT_DIM = 256
NUM_CLASSES = 38
BATCH_SIZE = 32

# Vision Transformer parameters
PATCH_SIZE = 16  # Divide latent vector into small patches
NUM_PATCHES = LATENT_DIM // PATCH_SIZE
NUM_TRANSFORMER_LAYERS = 4
NUM_HEADS = 4
DROPOUT_RATE = 0.1

# Training parameters
LEARNING_RATE = 5e-5
VAE_EPOCHS = 10
HYBRID_EPOCHS = 10
KL_WEIGHT = 0.0001  # Weight factor for KL loss in VAE (reduced from 0.001)
KL_ANNEALING_EPOCHS = 10  # Number of epochs to gradually increase KL weight

# Paths for saving models and checkpoints
VAE_WEIGHTS_PATH = "saved_models/vae"
HYBRID_WEIGHTS_PATH = "saved_models/hybrid"
VAE_CHECKPOINTS_PATH = "checkpoints/vae"
HYBRID_CHECKPOINTS_PATH = "checkpoints/hybrid"
