"""
Batch inference script for processing multiple images in PyTorch.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import time
import pandas as pd
from tqdm import tqdm
from torchvision import transforms

from utils.model_utils import load_trained_combined_model, get_device
from data.data_loader import get_class_mapping
from config.model_config import IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD

def predict_batch(image_paths, model, class_indices, device, batch_size=32):
    """
    Predict disease for a batch of images.

    Args:
        image_paths: List of image paths
        model: Loaded hybrid model
        class_indices: Dictionary mapping class names to indices
        device: Device to use for inference
        batch_size: Batch size for inference

    Returns:
        list: List of dictionaries with prediction results
    """
    # Define transform
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),  # Ensure exact size
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])

    # Initialize results
    results = []

    # Get class mapping
    idx_to_class = {v: k for k, v in class_indices.items()}

    # Set model to evaluation mode
    model.eval()

    # Process images in batches
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        valid_indices = []

        # Load and preprocess images
        for j, path in enumerate(batch_paths):
            try:
                img = Image.open(path).convert('RGB')
                img_tensor = transform(img)
                batch_images.append(img_tensor)
                valid_indices.append(j)
            except Exception as e:
                print(f"Error loading image {path}: {e}")

        if not batch_images:
            continue

        # Stack tensors
        batch_tensor = torch.stack(batch_images).to(device)

        # Measure inference time
        start_time = time.time()

        # Forward pass
        with torch.no_grad():
            # Get outputs from the combined model
            _, outputs, _, _, _ = model(batch_tensor)

        # Calculate inference time
        inference_time = (time.time() - start_time) / len(batch_images)

        # Process predictions
        probabilities = outputs.cpu().numpy()

        for j, idx in enumerate(valid_indices):
            path = batch_paths[idx]
            probs = probabilities[j]
            predicted_idx = np.argmax(probs)
            confidence = probs[predicted_idx]
            predicted_class = idx_to_class.get(predicted_idx, "Unknown")

            results.append({
                'image_path': path,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'inference_time_ms': inference_time * 1000
            })

    return results

def visualize_batch_results(results, output_dir):
    """
    Visualize batch results.

    Args:
        results: List of dictionaries with prediction results
        output_dir: Directory to save visualizations
    """
    # Create visualization directory
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    # Create a summary plot
    plt.figure(figsize=(12, 6))

    # Plot confidence distribution
    confidences = [r['confidence'] for r in results]
    plt.subplot(1, 2, 1)
    plt.hist(confidences, bins=20)
    plt.title('Confidence Distribution')
    plt.xlabel('Confidence')
    plt.ylabel('Count')

    # Plot inference time distribution
    inference_times = [r['inference_time_ms'] for r in results]
    plt.subplot(1, 2, 2)
    plt.hist(inference_times, bins=20)
    plt.title('Inference Time Distribution')
    plt.xlabel('Inference Time (ms)')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'batch_summary.png'))

    # Visualize individual predictions (limit to first 20)
    for i, result in enumerate(results[:20]):
        try:
            img = Image.open(result['image_path']).convert('RGB')
            plt.figure(figsize=(8, 6))
            plt.imshow(img)
            plt.title(f"Prediction: {result['predicted_class']}\nConfidence: {result['confidence']:.2%}")
            plt.axis('off')

            # Extract filename from path
            filename = os.path.basename(result['image_path'])
            base_name = os.path.splitext(filename)[0]

            plt.savefig(os.path.join(vis_dir, f"{base_name}_prediction.png"), bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error visualizing {result['image_path']}: {e}")

def main():
    """Main function for batch inference."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Batch inference for plant disease classification')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--visualize', action='store_true', help='Visualize the predictions')
    parser.add_argument('--model', type=str, default='combined', choices=['combined'],
                        help='Model type to use for inference (combined model)')
    args = parser.parse_args()

    # Set device
    device = get_device()
    print(f"Using device: {device}")

    # Load trained model
    print(f"Loading trained combined model...")
    model = load_trained_combined_model()

    if model is None:
        print("Failed to load combined model. Please train the model first.")
        return

    # Get image paths
    image_paths = []
    for filename in os.listdir(args.input_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff')):
            image_paths.append(os.path.join(args.input_dir, filename))

    if not image_paths:
        print(f"No images found in {args.input_dir}")
        return

    print(f"Found {len(image_paths)} images")

    # Get class indices
    class_indices = get_class_mapping()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Start timing
    start_time = time.time()

    # Process images
    print("Processing images...")
    results = predict_batch(image_paths, model, class_indices, device, args.batch_size)

    # Calculate total time
    total_time = time.time() - start_time
    avg_time_per_image = total_time / len(results) if results else 0

    # Save results to CSV
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, 'batch_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # Print summary
    print(f"Processed {len(results)} images in {total_time:.2f} seconds")
    print(f"Average time per image: {avg_time_per_image*1000:.2f} ms")

    # Visualize if requested
    if args.visualize:
        print("Generating visualizations...")
        visualize_batch_results(results, args.output_dir)
        print(f"Visualizations saved to {os.path.join(args.output_dir, 'visualizations')}")

if __name__ == "__main__":
    main()
