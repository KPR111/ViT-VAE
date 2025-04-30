"""
Inference script for the hybrid VAE-ViT model in PyTorch.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import time
from torchvision import transforms

from utils.model_utils import load_trained_hybrid_model, get_device
from data.data_loader import get_class_mapping
from config.model_config import IMAGE_SIZE

def predict_image(image_path, model, class_indices, device):
    """
    Predict disease for a single image.

    Args:
        image_path: Path to the image file
        model: Loaded hybrid model
        class_indices: Dictionary mapping class names to indices
        device: Device to use for inference

    Returns:
        tuple: (predicted_class, confidence, img, inference_time)
    """
    # Define transform
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),  # Ensure exact size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load and preprocess the image
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None, None, None

    # Measure inference time
    start_time = time.time()

    # Set model to evaluation mode
    model.eval()

    # Forward pass
    with torch.no_grad():
        outputs = model(img_tensor)

    # Calculate inference time
    inference_time = time.time() - start_time

    # Get prediction
    probabilities = outputs.cpu().numpy()[0]
    predicted_idx = np.argmax(probabilities)
    confidence = probabilities[predicted_idx]

    # Get class name
    idx_to_class = {v: k for k, v in class_indices.items()}
    predicted_class = idx_to_class.get(predicted_idx, "Unknown")

    return predicted_class, confidence, img, inference_time

def visualize_prediction(img, predicted_class, confidence, save_path=None):
    """
    Visualize the prediction.

    Args:
        img: PIL Image
        predicted_class: Predicted class name
        confidence: Confidence score
        save_path: Path to save the visualization (optional)
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.title(f"Prediction: {predicted_class}\nConfidence: {confidence:.2%}")
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")

    plt.show()

def main():
    """Main function for inference."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Inference for plant disease classification')
    parser.add_argument('--image', type=str, required=True, help='Path to the image file')
    parser.add_argument('--visualize', action='store_true', help='Visualize the prediction')
    parser.add_argument('--output', type=str, help='Path to save the visualization')
    args = parser.parse_args()

    # Set device
    device = get_device()
    print(f"Using device: {device}")

    # Load trained model
    print("Loading trained model...")
    hybrid_model = load_trained_hybrid_model()
    if hybrid_model is None:
        print("Failed to load hybrid model. Please train the model first.")
        return

    # Get class indices
    class_indices = get_class_mapping()

    # Predict
    print(f"Predicting disease for image: {args.image}")
    predicted_class, confidence, img, inference_time = predict_image(
        args.image, hybrid_model, class_indices, device
    )

    if predicted_class is None:
        print("Prediction failed.")
        return

    # Print results
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Inference time: {inference_time*1000:.2f} ms")

    # Visualize if requested
    if args.visualize and img is not None:
        visualize_prediction(img, predicted_class, confidence, args.output)

if __name__ == "__main__":
    main()
