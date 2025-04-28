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
        # Try multiple possible paths for the weights
        possible_paths = [
            "saved_models/hybrid/hybrid_final.weights.h5",  # Path from directory listing
            "saved_models/hybrid_final_weights.h5",         # Original path
            "saved_models/hybrid_complete_model.keras"      # Complete model path
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

# Test loading
if __name__ == "__main__":
    model = load_trained_hybrid_model()
    if model:
        print("Model architecture:")
        model.summary()