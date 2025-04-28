"""
Utility functions for model loading and management.
"""
import os
import tensorflow as tf
from models.vae import build_encoder, build_decoder, VAE
from models.vit import build_vit_classifier
from models.hybrid import build_hybrid_model
from config.model_config import VAE_WEIGHTS_PATH, HYBRID_WEIGHTS_PATH

def verify_vae_weights():
    """
    Verify that all necessary VAE weights exist.
    
    Returns:
        bool: True if all weights exist, False otherwise
    """
    required_files = [
        os.path.join(VAE_WEIGHTS_PATH, "encoder.weights.h5"),
        os.path.join(VAE_WEIGHTS_PATH, "decoder.weights.h5"),
        os.path.join(VAE_WEIGHTS_PATH, "vae_complete.weights.h5")
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("Missing VAE weight files:")
        for f in missing_files:
            print(f"- {f}")
        return False
    return True

def load_vae_encoder():
    """
    Load just the encoder part of the VAE.
    
    Returns:
        Model: Loaded encoder model or None if loading fails
    """
    encoder = build_encoder()
    try:
        encoder_path = os.path.join(VAE_WEIGHTS_PATH, "encoder.weights.h5")
        encoder.load_weights(encoder_path)
        print("Successfully loaded VAE encoder weights")
        return encoder
    except Exception as e:
        print(f"Could not load VAE encoder weights: {e}")
        return None

def load_complete_vae():
    """
    Load the complete VAE model.
    
    Returns:
        tuple: (vae, encoder, decoder) or (None, None, None) if loading fails
    """
    encoder = build_encoder()
    decoder = build_decoder()
    vae = VAE(encoder, decoder)
    
    try:
        vae_path = os.path.join(VAE_WEIGHTS_PATH, "vae_complete.weights.h5")
        vae.load_weights(vae_path)
        print("Successfully loaded complete VAE weights")
        return vae, encoder, decoder
    except Exception as e:
        print(f"Could not load complete VAE weights: {e}")
        return None, None, None

def load_trained_hybrid_model():
    """
    Load the trained hybrid model.
    
    Returns:
        Model: Loaded hybrid model or None if loading fails
    """
    # Build the base models
    encoder = build_encoder()
    vit_classifier = build_vit_classifier()
    
    # Build the hybrid model
    hybrid_model = build_hybrid_model(encoder, vit_classifier)
    
    # Compile the model (needed for loading weights)
    hybrid_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Try to load the weights
    try:
        # Try multiple possible paths for the weights
        possible_paths = [
            os.path.join(HYBRID_WEIGHTS_PATH, "hybrid_final.weights.h5"),
            "saved_models/hybrid_final_weights.h5",  # Legacy path
            os.path.join(HYBRID_WEIGHTS_PATH, "hybrid_complete_model.keras")
        ]
        
        for path in possible_paths:
            try:
                print(f"Trying to load weights from: {path}")
                hybrid_model.load_weights(path)
                print(f"Successfully loaded hybrid model weights from {path}")
                return hybrid_model
            except Exception as e:
                print(f"Could not load from {path}: {e}")
                continue
                
        # If we get here, none of the paths worked
        print("Could not find model weights in any of the expected locations")
        return None
    except Exception as e:
        print(f"Unexpected error loading model: {e}")
        return None

# Test loading when this file is run directly
if __name__ == "__main__":
    print("Testing model loading utilities...")
    
    print("\nVerifying VAE weights...")
    vae_weights_exist = verify_vae_weights()
    print(f"VAE weights exist: {vae_weights_exist}")
    
    if vae_weights_exist:
        print("\nLoading VAE encoder...")
        encoder = load_vae_encoder()
        
        print("\nLoading complete VAE...")
        vae, _, _ = load_complete_vae()
        
        print("\nLoading hybrid model...")
        hybrid_model = load_trained_hybrid_model()
        
        if hybrid_model:
            print("\nHybrid model architecture:")
            hybrid_model.summary()
