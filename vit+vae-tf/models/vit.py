"""
Vision Transformer (ViT) model implementation.
"""
from tensorflow.keras import layers, Model, Input
from config.model_config import (
    LATENT_DIM, NUM_CLASSES, PATCH_SIZE, NUM_PATCHES,
    NUM_TRANSFORMER_LAYERS, NUM_HEADS, DROPOUT_RATE
)

def mlp(x, hidden_units, dropout_rate):
    """
    Multilayer Perceptron block for Vision Transformer.
    
    Args:
        x: Input tensor
        hidden_units: List of hidden unit sizes
        dropout_rate: Dropout rate
        
    Returns:
        Tensor: Output tensor
    """
    for units in hidden_units:
        x = layers.Dense(units, activation='gelu')(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def build_vit_classifier():
    """
    Build the Vision Transformer classifier model.
    
    Returns:
        Model: ViT classifier model
    """
    inputs = Input(shape=(LATENT_DIM,), name="input_layer_vit")  # latent vector from VAE

    # Expand a little
    x = layers.Dense(256)(inputs)  # (optional, better performance)

    # Patch Embedding (split into patches)
    x = layers.Reshape((NUM_PATCHES, PATCH_SIZE))(x)

    # Transformer Blocks
    for _ in range(NUM_TRANSFORMER_LAYERS):
        # Layer Normalization
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)

        # Multi-Head Self Attention
        attention_output = layers.MultiHeadAttention(
            num_heads=NUM_HEADS, key_dim=PATCH_SIZE, dropout=DROPOUT_RATE
        )(x1, x1)

        # Skip Connection
        x2 = layers.Add()([attention_output, x])

        # Layer Normalization
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

        # MLP block
        x3 = mlp(x3, hidden_units=[PATCH_SIZE], dropout_rate=DROPOUT_RATE)

        # Skip Connection
        x = layers.Add()([x3, x2])

    # Final
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(DROPOUT_RATE * 3)(x)  # Higher dropout at the end
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs, name="vit_classifier")
    return model

# Only build and test model when this file is run directly
if __name__ == "__main__":
    # Build
    vit_classifier = build_vit_classifier()
    print("Vision Transformer Classifier Summary:")
    vit_classifier.summary()
