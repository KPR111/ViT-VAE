"""
Hybrid model that combines VAE and ViT for plant disease classification.
"""
from tensorflow.keras import Model, Input
import tensorflow as tf
from config.model_config import IMAGE_SHAPE

def build_hybrid_model(encoder, vit_classifier):
    """
    Build the hybrid model that combines VAE encoder and ViT classifier.
    
    Args:
        encoder: VAE encoder model
        vit_classifier: Vision Transformer classifier model
        
    Returns:
        Model: Hybrid model
    """
    inputs = Input(shape=IMAGE_SHAPE, name="input_layer_hybrid")

    # Get latent vector from encoder (we only need the z vector)
    z_mean, z_log_var, z = encoder(inputs)

    # Pass through Vision Transformer
    outputs = vit_classifier(z)

    model = Model(inputs, outputs, name="vae_vit_hybrid")
    return model

# Only build and test model when this file is run directly
if __name__ == "__main__":
    from vae import build_encoder
    from vit import build_vit_classifier
    
    # Build the base models
    encoder = build_encoder()
    vit_classifier = build_vit_classifier()
    
    # Build the hybrid model
    hybrid_model = build_hybrid_model(encoder, vit_classifier)
    
    # Print model summary
    print("Hybrid Model Summary:")
    hybrid_model.summary()
