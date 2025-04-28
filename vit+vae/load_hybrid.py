import tensorflow as tf
from models.vae_models import build_encoder
from models.vit_models import build_vit_classifier
from hybrid_model import build_hybrid_model

def load_trained_hybrid_model():
    # Build the base models
    encoder = build_encoder()
    vit_classifier = build_vit_classifier()
    
    # Build the hybrid model
    hybrid_model = build_hybrid_model(encoder, vit_classifier)
    
    # Load the trained weights
    try:
        hybrid_model.load_weights("saved_models/hybrid_final_weights.h5")
        print("Successfully loaded hybrid model weights")
    except:
        print("Could not load hybrid model weights")
        return None
    
    return hybrid_model

# Test loading
if __name__ == "__main__":
    model = load_trained_hybrid_model()
    if model:
        print("Model architecture:")
        model.summary()