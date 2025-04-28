import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from load_hybrid import load_trained_hybrid_model
from loadingpreprocessing import train_generator  # To get class indices
import argparse

def predict_image(image_path, model, class_indices):
    """Predict disease for a single image"""
    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make prediction
    prediction = model.predict(img_array)
    predicted_class_idx = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class_idx] * 100
    
    # Get class name
    idx_to_class = {v: k for k, v in class_indices.items()}
    predicted_class = idx_to_class[predicted_class_idx]
    
    return predicted_class, confidence, img

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test a single image with the trained model')
    parser.add_argument('--image', type=str, help='Path to the image file')
    args = parser.parse_args()
    
    # If no image path provided, use a default one
    if args.image:
        image_path = args.image
    else:
        # Use the first image from the test directory as default
        TEST_DIR = r"C:\Users\Prudvi\Downloads\archive\test\test"
        image_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not image_files:
            print("No image files found in the test directory")
            return
        image_path = os.path.join(TEST_DIR, image_files[0])
        print(f"No image specified, using: {image_path}")
    
    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return
    
    print("Loading the trained hybrid model...")
    # Load the trained hybrid model
    hybrid_model = load_trained_hybrid_model()
    
    if hybrid_model is None:
        print("Failed to load the hybrid model. Make sure you have trained the model first.")
        return
    
    # Get class indices from training data
    class_indices = train_generator.class_indices
    
    # Make prediction
    predicted_class, confidence, img = predict_image(image_path, hybrid_model, class_indices)
    
    # Display results
    print("\nPrediction Results:")
    print("-" * 50)
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    
    # Display the image with prediction
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}\nConfidence: {confidence:.2f}%")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Save the result
    os.makedirs("test_results", exist_ok=True)
    plt.savefig(f"test_results/single_prediction_{os.path.basename(image_path)}.png")
    print(f"Result saved to 'test_results/single_prediction_{os.path.basename(image_path)}.png'")

if __name__ == "__main__":
    main()
