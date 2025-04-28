import tensorflow as tf
from models.vae_models import build_encoder, build_decoder
# Import the VAE class directly to avoid circular imports
from trainvae import VAE

def load_vae_encoder():
    """Load just the encoder part of the VAE"""
    encoder = build_encoder()
    try:
        encoder.load_weights("saved_models/vae/encoder.weights.h5")  # Updated path
        print("Successfully loaded VAE encoder weights")
        return encoder
    except:
        print("Could not load VAE encoder weights")
        return None

def load_complete_vae():
    """Load the complete VAE model"""
    encoder = build_encoder()
    decoder = build_decoder()
    vae = VAE(encoder, decoder)

    try:
        vae.load_weights("saved_models/vae/vae_complete.weights.h5")  # Updated path
        print("Successfully loaded complete VAE weights")
        return vae, encoder, decoder
    except:
        print("Could not load complete VAE weights")
        return None, None, None

def verify_vae_weights():
    """Verify that all necessary VAE weights exist"""
    import os

    required_files = [
        "saved_models/vae/encoder.weights.h5",  # Updated paths
        "saved_models/vae/decoder.weights.h5",
        "saved_models/vae/vae_complete.weights.h5"
    ]

    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print("Missing VAE weight files:")
        for f in missing_files:
            print(f"- {f}")
        return False
    return True

