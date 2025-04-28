"""
Visualization utilities for model outputs and training history.
"""
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_history(history, save_path=None):
    """
    Plot training history metrics.
    
    Args:
        history: Training history object or dictionary
        save_path: Path to save the plot (optional)
    """
    # Convert to dictionary if it's a History object
    if hasattr(history, 'history'):
        history = history.history
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracy if available
    if 'accuracy' in history:
        axes[1].plot(history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history:
            axes[1].plot(history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_title('Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
    # Plot reconstruction loss for VAE
    elif 'reconstruction_loss' in history:
        axes[1].plot(history['reconstruction_loss'], label='Reconstruction Loss')
        axes[1].plot(history['kl_loss'], label='KL Loss')
        axes[1].set_title('VAE Losses')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    
    plt.show()

def visualize_predictions(images, true_classes, pred_classes, confidences, save_dir=None):
    """
    Visualize and save prediction results.
    
    Args:
        images: List of images
        true_classes: List of true class names
        pred_classes: List of predicted class names
        confidences: List of confidence scores
        save_dir: Directory to save visualizations (optional)
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Create a figure with subplots
    n_images = min(len(images), 16)  # Display up to 16 images
    cols = 4
    rows = (n_images + cols - 1) // cols
    
    plt.figure(figsize=(15, 4 * rows))
    
    for i in range(n_images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i])
        
        # Color based on correctness
        color = 'green' if true_classes[i] == pred_classes[i] else 'red'
        
        plt.title(f"True: {true_classes[i]}\nPred: {pred_classes[i]}\nConf: {confidences[i]:.1f}%", 
                 color=color, fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, "prediction_results.png"))
        print(f"Prediction visualization saved to {save_dir}/prediction_results.png")
    
    plt.show()

def visualize_reconstructions(original_images, reconstructed_images, n=10, save_path=None):
    """
    Visualize original images and their VAE reconstructions.
    
    Args:
        original_images: Original input images
        reconstructed_images: Reconstructed images from VAE
        n: Number of images to display
        save_path: Path to save the visualization (optional)
    """
    plt.figure(figsize=(20, 4))
    
    for i in range(n):
        # Display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original_images[i])
        plt.title("Original")
        plt.axis("off")
        
        # Display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed_images[i])
        plt.title("Reconstructed")
        plt.axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Reconstruction visualization saved to {save_path}")
    
    plt.show()

# Test visualizations when this file is run directly
if __name__ == "__main__":
    # Create dummy data for testing
    import numpy as np
    
    # Test training history plot
    history = {
        'loss': np.random.rand(10) * 0.5,
        'val_loss': np.random.rand(10) * 0.7,
        'accuracy': 0.7 + np.random.rand(10) * 0.2,
        'val_accuracy': 0.6 + np.random.rand(10) * 0.2
    }
    
    print("Testing training history visualization...")
    plot_training_history(history)
    
    # Test prediction visualization
    images = [np.random.rand(224, 224, 3) for _ in range(8)]
    true_classes = [f"Class_{i}" for i in range(8)]
    pred_classes = [f"Class_{i}" if i % 3 != 0 else f"Class_{i+1}" for i in range(8)]
    confidences = [80 + np.random.rand() * 20 for _ in range(8)]
    
    print("\nTesting prediction visualization...")
    visualize_predictions(images, true_classes, pred_classes, confidences)
