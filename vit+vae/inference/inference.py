"""
Inference script for the hybrid VAE-ViT model.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import argparse
import time

from utils.model_utils import load_trained_hybrid_model
from data.data_loader import get_class_mapping
from config.model_config import IMAGE_SHAPE

def predict_image(image_path, model, class_indices):
    """
    Predict disease for a single image.
    
    Args:
        image_path: Path to the image file
        model: Loaded hybrid model
        class_indices: Dictionary mapping class names to indices
        
    Returns:
        tuple: (predicted_class, confidence, img, inference_time)
    """
    # Load and preprocess the image
    try:
        img = load_img(image_path, target_size=IMAGE_SHAPE[:2])
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
    """Main function for inference."""
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
    hybrid_model = load_trained_hybrid_model()
    
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
