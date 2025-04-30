"""
Batch inference script for processing multiple images.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import argparse
import time
import pandas as pd
from tqdm import tqdm

from utils.model_utils import load_trained_hybrid_model
from data.data_loader import get_class_mapping
from config.model_config import IMAGE_SHAPE

def predict_batch(image_paths, model, class_indices, batch_size=32):
    """
    Predict disease for a batch of images.
    
    Args:
        image_paths: List of image paths
        model: Loaded hybrid model
        class_indices: Dictionary mapping class names to indices
        batch_size: Batch size for inference
        
    Returns:
        list: List of dictionaries with prediction results
    """
    # Initialize results
    results = []
    
    # Process images in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        valid_indices = []
        
        # Load and preprocess images
        for j, path in enumerate(batch_paths):
            try:
                img = load_img(path, target_size=IMAGE_SHAPE[:2])
                img_array = img_to_array(img) / 255.0
                batch_images.append(img_array)
                valid_indices.append(j)
            except Exception as e:
                print(f"Error loading image {path}: {e}")
        
        if not batch_images:
            continue
            
        # Convert to numpy array
        batch_images = np.array(batch_images)
        
        # Make predictions
        batch_predictions = model.predict(batch_images, verbose=0)
        
        # Process predictions
        idx_to_class = {v: k for k, v in class_indices.items()}
        
        for j, pred in enumerate(batch_predictions):
            pred_class_idx = np.argmax(pred)
            confidence = pred[pred_class_idx] * 100
            pred_class = idx_to_class[pred_class_idx]
            
            # Get original path
            original_idx = valid_indices[j]
            path = batch_paths[original_idx]
            
            results.append({
                'image_path': path,
                'predicted_class': pred_class,
                'confidence': confidence
            })
    
    return results

def main():
    """Main function for batch inference."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Batch inference on plant leaf images')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--output_dir', type=str, default='batch_results', help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization for each image')
    args = parser.parse_args()
    
    # Check if the input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        return
    
    # Get all image files
    image_paths = []
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                image_paths.append(os.path.join(root, file))
    
    if not image_paths:
        print(f"No image files found in {args.input_dir}")
        return
    
    print(f"Found {len(image_paths)} images")
    
    print("Loading the trained hybrid model...")
    # Load the trained hybrid model
    hybrid_model = load_trained_hybrid_model()
    
    if hybrid_model is None:
        print("Failed to load the hybrid model. Make sure you have trained the model first.")
        return
    
    # Get class indices
    class_indices = get_class_mapping()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start timing
    start_time = time.time()
    
    # Process images
    print("Processing images...")
    results = predict_batch(image_paths, hybrid_model, class_indices, args.batch_size)
    
    # Calculate total time
    total_time = time.time() - start_time
    avg_time_per_image = total_time / len(results) if results else 0
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, 'batch_results.csv')
    results_df.to_csv(csv_path, index=False)
    
    # Print summary
    print("\nBatch Processing Results:")
    print("-" * 50)
    print(f"Total images processed: {len(results)}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per image: {avg_time_per_image:.4f} seconds")
    print(f"Results saved to: {csv_path}")
    
    # Generate visualizations if requested
    if args.visualize:
        print("Generating visualizations...")
        vis_dir = os.path.join(args.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        for i, result in enumerate(tqdm(results)):
            try:
                # Load image
                img = plt.imread(result['image_path'])
                
                # Create figure
                plt.figure(figsize=(8, 8))
                plt.imshow(img)
                plt.title(f"Predicted: {result['predicted_class']}\nConfidence: {result['confidence']:.2f}%")
                plt.axis('off')
                plt.tight_layout()
                
                # Save figure
                output_filename = f"prediction_{i}_{os.path.basename(result['image_path'])}"
                output_path = os.path.join(vis_dir, output_filename)
                plt.savefig(output_path)
                plt.close()
            except Exception as e:
                print(f"Error generating visualization for {result['image_path']}: {e}")
        
        print(f"Visualizations saved to: {vis_dir}")
    
    print("Done!")

if __name__ == "__main__":
    main()
