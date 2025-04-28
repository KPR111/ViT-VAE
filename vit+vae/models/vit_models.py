from tensorflow.keras import layers, Model, Input

# Constants
NUM_CLASSES = 38
LATENT_DIM = 256
PATCH_SIZE = 16  # Divide latent vector into small patches
NUM_PATCHES = LATENT_DIM // PATCH_SIZE

# ViT Patch + Transformer block
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation='gelu')(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

# Vision Transformer Model
def build_vit_classifier():
    inputs = Input(shape=(LATENT_DIM,))  # latent vector from VAE

    # Expand a little
    x = layers.Dense(256)(inputs)  # (optional, better performance)

    # Patch Embedding (split into patches)
    x = layers.Reshape((NUM_PATCHES, PATCH_SIZE))(x)

    # Transformer Block
    for _ in range(4):  # 4 transformer encoder layers
        # Layer Normalization
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)

        # Multi-Head Self Attention
        attention_output = layers.MultiHeadAttention(
            num_heads=4, key_dim=PATCH_SIZE, dropout=0.1
        )(x1, x1)

        # Skip Connection
        x2 = layers.Add()([attention_output, x])

        # Layer Normalization
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

        # MLP block
        x3 = mlp(x3, hidden_units=[PATCH_SIZE], dropout_rate=0.1)

        # Skip Connection
        x = layers.Add()([x3, x2])

    # Final
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs, name="vit_classifier")
    return model

# Only build and test model when this file is run directly
if __name__ == "__main__":
    # Build
    vit_classifier = build_vit_classifier()
    print("Vision Transformer Classifier Summary:")
    vit_classifier.summary()
