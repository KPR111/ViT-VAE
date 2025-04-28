# Technical Documentation: VAE-ViT Hybrid Model

This document provides technical details about the implementation, usage, and structure of the VAE-ViT hybrid model for plant disease classification.

## Project Structure

```
vit+vae/
├── checkpoints/             # Saved model checkpoints during training
│   ├── hybrid/              # Hybrid model checkpoints
│   └── vae/                 # VAE model checkpoints
├── models/                  # Model definitions
│   ├── vae_models.py        # VAE encoder and decoder architecture
│   └── vit_models.py        # Vision Transformer architecture
├── saved_models/            # Final trained models
│   ├── hybrid/              # Hybrid model weights
│   └── vae/                 # VAE model weights
├── conda.txt                # Conda environment specification
├── gpu_setup.py             # GPU configuration for TensorFlow
├── hybrid_model.py          # Hybrid model definition and training
├── load_hybrid.py           # Utility to load trained hybrid model
├── load_vae.py              # Utility to load trained VAE components
├── loadingpreprocessing.py  # Data loading and preprocessing
├── plot.py                  # Visualization utilities
├── requirements.txt         # Python package dependencies
└── trainvae.py              # VAE training script
```

## Installation and Setup

### Environment Setup

1. Create a conda environment using the provided specification:
   ```
   conda create --name vae-vit --file conda.txt
   conda activate vae-vit
   ```

2. Install additional dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure GPU settings (if available):
   ```
   python gpu_setup.py
   ```

## Data Preparation

The project uses the "New Plant Diseases Dataset" which should be organized as follows:
- Training data: `/path/to/New Plant Diseases Dataset(Augmented)/train/`
- Validation data: `/path/to/New Plant Diseases Dataset(Augmented)/valid/`

Update the paths in `loadingpreprocessing.py` to point to your dataset location.

## Training Pipeline

### Step 1: Train the VAE

```
python trainvae.py
```

This script:
1. Builds the VAE encoder and decoder
2. Trains the VAE to reconstruct plant leaf images
3. Saves the encoder and decoder weights separately
4. Saves checkpoints during training

Note: Adjust the number of epochs in the script before running (default is set to 1 for testing).

### Step 2: Train the Hybrid Model

```
python hybrid_model.py
```

This script:
1. Loads the pre-trained VAE encoder
2. Freezes the encoder weights
3. Builds the Vision Transformer classifier
4. Combines them into a hybrid model
5. Trains the hybrid model for plant disease classification
6. Saves the final model and weights

Note: Adjust the number of epochs in the script before running (default is set to 1 for testing).

## Model Architecture Details

### VAE Encoder

```python
def build_encoder():
    inputs = Input(shape=IMAGE_SHAPE)  # (224, 224, 3)

    x = layers.Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)

    z_mean = layers.Dense(LATENT_DIM, name='z_mean')(x)  # LATENT_DIM = 256
    z_log_var = layers.Dense(LATENT_DIM, name='z_log_var')(x)

    # Sampling function (reparameterization trick)
    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(sampling, output_shape=(LATENT_DIM,), name='z')([z_mean, z_log_var])

    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    return encoder
```

### Vision Transformer

```python
def build_vit_classifier():
    inputs = Input(shape=(LATENT_DIM,))  # LATENT_DIM = 256

    # Expand a little
    x = layers.Dense(256)(inputs)

    # Patch Embedding (split into patches)
    x = layers.Reshape((NUM_PATCHES, PATCH_SIZE))(x)  # NUM_PATCHES = 16, PATCH_SIZE = 16

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
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)  # NUM_CLASSES = 38

    model = Model(inputs=inputs, outputs=outputs, name="vit_classifier")
    return model
```

### Hybrid Model

```python
def build_hybrid_model(encoder, vit_classifier):
    inputs = Input(shape=(224, 224, 3))

    # Get latent vector from encoder (we only need the z vector)
    z_mean, z_log_var, z = encoder(inputs)

    # Pass through Vision Transformer
    outputs = vit_classifier(z)

    model = Model(inputs, outputs, name="vae_vit_hybrid")
    return model
```

## Using the Trained Models

### Loading the VAE

```python
from load_vae import load_complete_vae

vae, encoder, decoder = load_complete_vae()
```

### Loading the Hybrid Model

```python
from load_hybrid import load_trained_hybrid_model

hybrid_model = load_trained_hybrid_model()
```

### Making Predictions

```python
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load and preprocess an image
img = load_img('path/to/image.jpg', target_size=(224, 224))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Make prediction
predictions = hybrid_model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)[0]

# Map to class name
class_names = list(train_generator.class_indices.keys())
predicted_class_name = class_names[predicted_class]
print(f"Predicted disease: {predicted_class_name}")
```

## Performance Metrics

To evaluate the model's performance, you can use:

```python
# Evaluate on validation data
evaluation = hybrid_model.evaluate(val_generator)
print(f"Loss: {evaluation[0]}, Accuracy: {evaluation[1]}")

# For more detailed metrics
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Get predictions
val_generator.reset()
y_pred = []
y_true = []

for i in range(len(val_generator)):
    x_batch, y_batch = val_generator[i]
    pred_batch = hybrid_model.predict(x_batch)
    y_pred.extend(np.argmax(pred_batch, axis=1))
    y_true.extend(np.argmax(y_batch, axis=1))

# Print classification report
class_names = list(val_generator.class_indices.keys())
print(classification_report(y_true, y_pred, target_names=class_names))

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)
```

## Computational Requirements

- **Training**: The model training is computationally intensive and benefits from GPU acceleration.
- **Memory**: At least 8GB of GPU memory is recommended for training with the default batch size.
- **Inference**: The trained model can run on CPU for inference, but GPU acceleration is recommended for batch processing.

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**:
   - Reduce batch size in `loadingpreprocessing.py`
   - Use a smaller input image size (requires adjusting the model architecture)

2. **VAE Training Instability**:
   - Adjust the balance between reconstruction loss and KL divergence
   - Try different learning rates

3. **Missing Weights Files**:
   - Ensure you've run `trainvae.py` before `hybrid_model.py`
   - Check that the paths in `load_vae.py` and `load_hybrid.py` match your directory structure

## Advanced Usage

### Fine-tuning the Hybrid Model

To fine-tune the entire model (including the VAE encoder):

```python
# In hybrid_model.py, change:
encoder.trainable = True  # Instead of False

# Use a lower learning rate
hybrid_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Lower learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### Visualizing Attention Maps

To visualize which parts of the latent representation the ViT is attending to:

```python
# Create a model that outputs attention weights
attention_model = tf.keras.Model(
    inputs=hybrid_model.input,
    outputs=[hybrid_model.get_layer('multi_head_attention').output]
)

# Get attention weights for an image
img = load_img('path/to/image.jpg', target_size=(224, 224))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)
attention_weights = attention_model.predict(img_array)

# Visualize
import matplotlib.pyplot as plt
plt.imshow(attention_weights[0])
plt.colorbar()
plt.title('Attention Map')
plt.show()
```

## References

- Vaswani, A., et al. (2017). "Attention is All You Need." NeurIPS 2017.
- Kingma, D. P., & Welling, M. (2013). "Auto-Encoding Variational Bayes." ICLR 2014.
- Dosovitskiy, A., et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR 2021.
