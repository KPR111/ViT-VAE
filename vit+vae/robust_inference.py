import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import argparse
import time

# Constants
IMAGE_SHAPE = (224, 224, 3)
LATENT_DIM = 256
NUM_CLASSES = 38
PATCH_SIZE = 16
NUM_PATCHES = LATENT_DIM // PATCH_SIZE

def build_encoder():
    """Rebuild the VAE encoder architecture"""
    inputs = Input(shape=IMAGE_SHAPE, name="input_layer")

    x = layers.Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)

    z_mean = layers.Dense(LATENT_DIM, name='z_mean')(x)
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

def mlp(x, hidden_units, dropout_rate):
    """MLP block for Vision Transformer"""
    for units in hidden_units:
        x = layers.Dense(units, activation='gelu')(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def build_vit_classifier():
    """Rebuild the Vision Transformer classifier architecture"""
    inputs = Input(shape=(LATENT_DIM,), name="input_layer_vit")  # latent vector from VAE

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

def build_hybrid_model(encoder, vit_classifier):
    """Build the hybrid model architecture"""
    inputs = Input(shape=IMAGE_SHAPE, name="input_layer_hybrid")

    # Get latent vector from encoder (we only need the z vector)
    z_mean, z_log_var, z = encoder(inputs)

    # Pass through Vision Transformer
    outputs = vit_classifier(z)

    model = Model(inputs, outputs, name="vae_vit_hybrid")
    return model

def load_model():
    """Load the trained hybrid model by rebuilding the architecture and loading weights"""
    print("Building model architecture...")
    
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
        print("Trying to load weights from: saved_models/hybrid/hybrid_final.weights.h5")
        hybrid_model.load_weights("saved_models/hybrid/hybrid_final.weights.h5")
        print("Successfully loaded hybrid model weights")
        return hybrid_model
    except Exception as e:
        print(f"Could not load weights: {e}")
        return None

def get_class_mapping():
    """Get the mapping between class indices and class names"""
    # This is a hardcoded version of the class mapping to avoid importing train_generator
    class_indices = {
        'Apple___Apple_scab': 0, 
        'Apple___Black_rot': 1, 
        'Apple___Cedar_apple_rust': 2, 
        'Apple___healthy': 3, 
        'Blueberry___healthy': 4, 
        'Cherry_(including_sour)___Powdery_mildew': 5, 
        'Cherry_(including_sour)___healthy': 6, 
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 7, 
        'Corn_(maize)___Common_rust_': 8, 
        'Corn_(maize)___Northern_Leaf_Blight': 9, 
        'Corn_(maize)___healthy': 10, 
        'Grape___Black_rot': 11, 
        'Grape___Esca_(Black_Measles)': 12, 
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 13, 
        'Grape___healthy': 14, 
        'Orange___Haunglongbing_(Citrus_greening)': 15, 
        'Peach___Bacterial_spot': 16, 
        'Peach___healthy': 17, 
        'Pepper,_bell___Bacterial_spot': 18, 
        'Pepper,_bell___healthy': 19, 
        'Potato___Early_blight': 20, 
        'Potato___Late_blight': 21, 
        'Potato___healthy': 22, 
        'Raspberry___healthy': 23, 
        'Soybean___healthy': 24, 
        'Squash___Powdery_mildew': 25, 
        'Strawberry___Leaf_scorch': 26, 
        'Strawberry___healthy': 27, 
        'Tomato___Bacterial_spot': 28, 
        'Tomato___Early_blight': 29, 
        'Tomato___Late_blight': 30, 
        'Tomato___Leaf_Mold': 31, 
        'Tomato___Septoria_leaf_spot': 32, 
        'Tomato___Spider_mites Two-spotted_spider_mite': 33, 
        'Tomato___Target_Spot': 34, 
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 35, 
        'Tomato___Tomato_mosaic_virus': 36, 
        'Tomato___healthy': 37
    }
    return class_indices

def predict_image(image_path, model, class_indices):
    """Predict disease for a single image"""
    # Load and preprocess the image
    try:
        img = load_img(image_path, target_size=(224, 224))
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None, None, None
        
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make prediction
    start_time = time.time()
    prediction = model.predict(img_array, verbose=0)  # Set verbose=0 to reduce output
    inference_time = time.time() - start_time
    
    predicted_class_idx = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class_idx] * 100
    
    # Get class name
    idx_to_class = {v: k for k, v in class_indices.items()}
    predicted_class = idx_to_class[predicted_class_idx]
    
    return predicted_class, confidence, img, inference_time

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Make inference on a plant leaf image')
    parser.add_argument('--image', type=str, required=True, help='Path to the image file')
    parser.add_argument('--save_dir', type=str, default='inference_results', help='Directory to save results')
    args = parser.parse_args()
    
    # Check if the image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    print("Loading the trained hybrid model...")
    # Load the trained hybrid model
    hybrid_model = load_model()
    
    if hybrid_model is None:
        print("Failed to load the hybrid model. Make sure you have trained the model first.")
        return
    
    # Get class indices
    class_indices = get_class_mapping()
    
    # Make prediction
    predicted_class, confidence, img, inference_time = predict_image(args.image, hybrid_model, class_indices)
    
    if predicted_class is None:
        return
    
    # Display results
    print("\nPrediction Results:")
    print("-" * 50)
    print(f"Image: {os.path.basename(args.image)}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Inference Time: {inference_time:.4f} seconds")
    
    # Create directory to save results
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Display the image with prediction
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}\nConfidence: {confidence:.2f}%")
    plt.axis('off')
    plt.tight_layout()
    
    # Save the result
    result_path = os.path.join(args.save_dir, f"prediction_{os.path.basename(args.image)}")
    plt.savefig(result_path)
    print(f"Result saved to '{result_path}'")
    
    # Show the plot (comment this out if running in a non-interactive environment)
    plt.show()

if __name__ == "__main__":
    main()
