"""
Utility functions for model loading and management in PyTorch.
"""
import os
import torch
from models.vae import build_encoder, build_decoder, VAE
from models.vit import build_vit_classifier
from models.hybrid import build_hybrid_model
from models.combined_model import build_combined_model
from config.model_config import VAE_WEIGHTS_PATH, HYBRID_WEIGHTS_PATH, DEVICE, BATCH_SIZE

def get_device():
    """
    Get the device to use for training and inference.

    Returns:
        torch.device: Device to use
    """
    return torch.device(DEVICE if torch.cuda.is_available() and DEVICE == "cuda" else "cpu")

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    Save a checkpoint of the model.

    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save the checkpoint
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)

    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(model, optimizer, filepath):
    """
    Load a checkpoint of the model.

    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        filepath: Path to the checkpoint

    Returns:
        tuple: (model, optimizer, epoch, loss)
    """
    if not os.path.exists(filepath):
        print(f"Checkpoint not found at {filepath}")
        return model, optimizer, 0, float('inf')

    # Load checkpoint
    checkpoint = torch.load(filepath, map_location=get_device())
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    print(f"Checkpoint loaded from {filepath} (epoch {epoch})")

    return model, optimizer, epoch, loss

def load_vae_encoder():
    """
    Load the VAE encoder model.

    Returns:
        Encoder: Loaded encoder model or None if loading fails
    """
    encoder = build_encoder()
    encoder = encoder.to(get_device())

    try:
        encoder_path = os.path.join(VAE_WEIGHTS_PATH, "encoder.pt")
        encoder.load_state_dict(torch.load(encoder_path, map_location=get_device()))
        print("Successfully loaded VAE encoder weights")
        return encoder
    except Exception as e:
        print(f"Could not load VAE encoder weights: {e}")
        return None

def load_vae_decoder():
    """
    Load the VAE decoder model.

    Returns:
        Decoder: Loaded decoder model or None if loading fails
    """
    decoder = build_decoder()
    decoder = decoder.to(get_device())

    try:
        decoder_path = os.path.join(VAE_WEIGHTS_PATH, "decoder.pt")
        decoder.load_state_dict(torch.load(decoder_path, map_location=get_device()))
        print("Successfully loaded VAE decoder weights")
        return decoder
    except Exception as e:
        print(f"Could not load VAE decoder weights: {e}")
        return None

def load_complete_vae():
    """
    Load the complete VAE model.

    Returns:
        tuple: (vae, encoder, decoder) or (None, None, None) if loading fails
    """
    encoder = build_encoder()
    decoder = build_decoder()
    vae = VAE()

    # Move models to device
    device = get_device()
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    vae = vae.to(device)

    try:
        vae_path = os.path.join(VAE_WEIGHTS_PATH, "vae_complete.pt")
        vae.load_state_dict(torch.load(vae_path, map_location=device))
        print("Successfully loaded complete VAE weights")
        return vae, encoder, decoder
    except Exception as e:
        print(f"Could not load complete VAE weights: {e}")
        return None, None, None

def verify_vae_weights():
    """
    Verify that the VAE weights exist.

    Returns:
        bool: True if weights exist, False otherwise
    """
    encoder_path = os.path.join(VAE_WEIGHTS_PATH, "encoder.pt")
    decoder_path = os.path.join(VAE_WEIGHTS_PATH, "decoder.pt")
    vae_path = os.path.join(VAE_WEIGHTS_PATH, "vae_complete.pt")

    return os.path.exists(encoder_path) and os.path.exists(decoder_path) and os.path.exists(vae_path)

def load_trained_hybrid_model():
    """
    Load the trained hybrid model.

    Returns:
        HybridModel: Loaded hybrid model or None if loading fails
    """
    # Load encoder
    encoder = load_vae_encoder()
    if encoder is None:
        return None

    # Build ViT classifier
    vit_classifier = build_vit_classifier()

    # Build hybrid model
    hybrid_model = build_hybrid_model(encoder, vit_classifier)

    # Move model to device
    device = get_device()
    hybrid_model = hybrid_model.to(device)

    try:
        hybrid_path = os.path.join(HYBRID_WEIGHTS_PATH, "hybrid_model.pt")
        hybrid_model.load_state_dict(torch.load(hybrid_path, map_location=device))
        print("Successfully loaded hybrid model weights")
        return hybrid_model
    except Exception as e:
        print(f"Could not load hybrid model weights: {e}")
        return None

def load_trained_combined_model():
    """
    Load the trained combined model.

    Returns:
        CombinedModel: Loaded combined model or None if loading fails
    """
    # Build the base models
    encoder = build_encoder()
    decoder = build_decoder()
    vit_classifier = build_vit_classifier()

    # Get the size of the training dataset (approximate value for loading)
    train_data_size = 1000  # This is just a placeholder value for loading

    # Build combined model
    combined_model = build_combined_model(
        encoder, decoder, vit_classifier,
        batch_size=BATCH_SIZE,
        train_data_size=train_data_size
    )

    # Move model to device
    device = get_device()
    combined_model = combined_model.to(device)

    try:
        combined_path = os.path.join(HYBRID_WEIGHTS_PATH, "combined_model.pt")
        combined_model.load_state_dict(torch.load(combined_path, map_location=device))
        print("Successfully loaded combined model weights")
        return combined_model
    except Exception as e:
        print(f"Could not load combined model weights: {e}")
        return None

# Only run tests when this file is run directly
if __name__ == "__main__":
    # Test device
    device = get_device()
    print(f"Using device: {device}")

    # Test directory creation
    os.makedirs(VAE_WEIGHTS_PATH, exist_ok=True)
    os.makedirs(HYBRID_WEIGHTS_PATH, exist_ok=True)
    print(f"Created directories: {VAE_WEIGHTS_PATH}, {HYBRID_WEIGHTS_PATH}")

    # Test model building
    encoder = build_encoder()
    decoder = build_decoder()
    vae = VAE()
    vit_classifier = build_vit_classifier()
    hybrid_model = build_hybrid_model(encoder, vit_classifier)

    # Test combined model building
    train_data_size = 1000  # Example value
    combined_model = build_combined_model(encoder, decoder, vit_classifier, BATCH_SIZE, train_data_size)

    print("Successfully built all models")
