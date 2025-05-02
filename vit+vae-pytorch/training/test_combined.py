"""
Testing script for the combined VAE-ViT model in PyTorch.
Evaluates the model on a test dataset and generates performance metrics.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from datetime import datetime

from data.data_loader import initialize_data_loaders
from data.test_data_loader import create_flat_test_loader
from utils.model_utils import load_trained_combined_model, get_device
from config.model_config import BATCH_SIZE, TEST_DIR, NORMALIZE_MEAN, NORMALIZE_STD

def create_test_loader(test_dir=TEST_DIR, batch_size=BATCH_SIZE):
    """
    Create a data loader for the test dataset.

    Args:
        test_dir (str): Path to the test directory
        batch_size (int): Batch size for the data loader

    Returns:
        tuple: (test_loader, class_to_idx)
    """
    try:
        # First, get the class mapping from the training data
        # This ensures consistent class indices between training and testing
        initialize_data_loaders(verbose=False)
        from data.data_loader import class_to_idx as train_class_to_idx

        # Create test loader with flat directory structure
        # where class names are extracted from filenames
        test_loader, test_class_to_idx = create_flat_test_loader(
            test_dir=test_dir,
            batch_size=batch_size,
            class_mapping=train_class_to_idx
        )

        # Check if dataset is empty
        if len(test_loader.dataset) == 0:
            print(f"No valid images found in test directory: {test_dir}")
            return test_loader, test_class_to_idx

        print(f"Created test loader with {len(test_loader.dataset)} images")
        print(f"Found {len(test_class_to_idx)} classes in test dataset")

        return test_loader, test_class_to_idx

    except Exception as e:
        print(f"Error creating test data loader: {e}")
        # Create an empty dataset with a minimal class_to_idx
        empty_dataset = torch.utils.data.TensorDataset(
            torch.tensor([]), torch.tensor([])
        )
        empty_loader = torch.utils.data.DataLoader(empty_dataset, batch_size=1)
        return empty_loader, {}

def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on the test dataset.

    Args:
        model: Trained combined model
        test_loader: Data loader for the test dataset
        device: Device to use for evaluation

    Returns:
        tuple: (accuracy, predictions, targets, class_probs)
    """
    model.eval()
    all_predictions = []
    all_targets = []
    all_class_probs = []

    with torch.no_grad():
        for data, targets in tqdm(test_loader, desc="Evaluating"):
            # Move data to device
            data = data.to(device)
            targets = targets.to(device)

            # Forward pass
            _, class_outputs, _, _, _ = model(data)  # We only need class outputs for evaluation

            # Get predictions
            _, predictions = class_outputs.max(1)

            # Store predictions and targets
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_class_probs.append(torch.softmax(class_outputs, dim=1).cpu().numpy())

    # Check if we have any predictions
    if not all_predictions:
        print("No samples were processed. Check if your test directory contains valid data.")
        return 0.0, [], [], []

    # Concatenate class probabilities if we have any
    if all_class_probs:
        all_class_probs = np.concatenate(all_class_probs, axis=0)
    else:
        all_class_probs = np.array([])

    # Calculate accuracy
    accuracy = accuracy_score(all_targets, all_predictions)

    return accuracy, all_predictions, all_targets, all_class_probs

def plot_confusion_matrix(y_true, y_pred, class_names, output_dir):
    """
    Plot and save the confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_dir: Directory to save the plot
    """
    # Check if we have any data
    if len(y_true) == 0 or len(y_pred) == 0:
        print("No data available for confusion matrix.")
        return

    try:
        # Get unique classes in the test dataset
        unique_classes = sorted(list(set(y_true)))

        # Get class names for only the classes that appear in the test dataset
        test_class_names = [class_names[i] for i in unique_classes]

        # Create confusion matrix only for classes that appear in the test dataset
        cm = confusion_matrix(y_true, y_pred, labels=unique_classes)

        # Plot confusion matrix
        plt.figure(figsize=(20, 16))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=test_class_names,
                   yticklabels=test_class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()

        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()
    except Exception as e:
        print(f"Error generating confusion matrix: {e}")
        return

def plot_reconstruction_samples(model, test_loader, device, output_dir, num_samples=5):
    """
    Plot and save sample reconstructions from the VAE component.

    Args:
        model: Trained combined model
        test_loader: Data loader for the test dataset
        device: Device to use for evaluation
        output_dir: Directory to save the plots
        num_samples: Number of samples to plot
    """
    model.eval()

    # Check if test_loader has any data
    if len(test_loader) == 0:
        print("No data available for reconstruction visualization.")
        return

    try:
        # Get a batch of test data
        data_iter = iter(test_loader)
        images, _ = next(data_iter)

        # Check if we have enough samples
        if len(images) == 0:
            print("No images available for reconstruction visualization.")
            return

        # Adjust num_samples if we don't have enough images
        if len(images) < num_samples:
            num_samples = len(images)
            print(f"Only {num_samples} images available for reconstruction visualization.")

        # Select a subset of images
        images = images[:num_samples].to(device)

        with torch.no_grad():
            # Get reconstructions
            reconstructed, _, _, _, _ = model(images)

        # Convert tensors to numpy arrays
        images = images.cpu().numpy()
        reconstructed = reconstructed.cpu().numpy()

        # Plot original and reconstructed images
        _, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    except Exception as e:
        print(f"Error generating reconstruction samples: {e}")
        return

    for i in range(num_samples):
        # Original images
        orig_img = np.transpose(images[i], (1, 2, 0))
        # Denormalize
        orig_img = orig_img * np.array(NORMALIZE_STD) + np.array(NORMALIZE_MEAN)
        orig_img = np.clip(orig_img, 0, 1)

        # Reconstructed images
        recon_img = np.transpose(reconstructed[i], (1, 2, 0))
        recon_img = np.clip(recon_img, 0, 1)

        # Plot
        axes[0, i].imshow(orig_img)
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')

        axes[1, i].imshow(recon_img)
        axes[1, i].set_title('Reconstructed')
        axes[1, i].axis('off')

    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'reconstruction_samples.png'))
    plt.close()

def plot_latent_space(model, test_loader, device, output_dir, num_samples=1000):
    """
    Plot and save a visualization of the latent space.
    Uses PCA to reduce the dimensionality of the latent space to 2D.

    Args:
        model: Trained combined model
        test_loader: Data loader for the test dataset
        device: Device to use for evaluation
        output_dir: Directory to save the plot
        num_samples: Number of samples to use for the plot
    """
    from sklearn.decomposition import PCA

    # Check if test_loader has any data
    if len(test_loader) == 0:
        print("No data available for latent space visualization.")
        return

    try:
        model.eval()
        latent_vectors = []
        labels = []
        count = 0

        with torch.no_grad():
            for data, targets in test_loader:
                # Move data to device
                data = data.to(device)

                # Get latent vectors
                _, _, z = model.encoder(data)

                # Store latent vectors and labels
                latent_vectors.append(z.cpu().numpy())
                labels.append(targets.numpy())

                # Update count
                count += len(data)
                if count >= num_samples:
                    break

        # Check if we collected any data
        if not latent_vectors:
            print("No latent vectors collected for visualization.")
            return

        # Concatenate latent vectors and labels
        latent_vectors = np.concatenate(latent_vectors, axis=0)[:num_samples]
        labels = np.concatenate(labels, axis=0)[:num_samples]

        # Apply PCA
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(latent_vectors)

        # Plot latent space
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='tab20', alpha=0.7, s=10)
        plt.colorbar(scatter, label='Class')
        plt.title('PCA Visualization of Latent Space')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.tight_layout()
    except Exception as e:
        print(f"Error generating latent space visualization: {e}")
        return

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'latent_space.png'))
    plt.close()

def test_combined_model(test_dir=TEST_DIR, output_dir="test_results"):
    """
    Test the combined model on the test dataset and generate performance metrics.

    Args:
        test_dir (str): Path to the test directory
        output_dir (str): Directory to save the results

    Returns:
        dict: Dictionary containing test results
    """
    # Check if test directory exists
    if not os.path.exists(test_dir):
        print(f"Test directory '{test_dir}' does not exist.")
        print("Please provide a valid test directory path.")
        return None

    # Set device
    device = get_device()
    print(f"Using device: {device}")

    # Create timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Load trained model
    print("Loading trained combined model...")
    model = load_trained_combined_model()

    if model is None:
        print("Failed to load combined model. Please train the model first.")
        return None

    # Create test data loader
    print("Creating test data loader...")
    try:
        test_loader, class_to_idx = create_test_loader(test_dir)

        # Check if test directory exists and contains data
        if len(test_loader) == 0:
            print(f"Test directory '{test_dir}' exists but contains no valid data.")
            print(f"Please check that the directory contains properly formatted image files.")
            return None

        # Get class names
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

        # Evaluate model
        print("Evaluating model...")
        accuracy, predictions, targets, class_probs = evaluate_model(model, test_loader, device)

        # Check if we have any predictions
        if len(predictions) == 0:
            print("No predictions were made. Check your test data.")
            return None

        # Calculate precision, recall, and F1 score
        precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average='weighted')

        # Get unique classes in the test dataset
        unique_classes = sorted(list(set(targets)))

        # Get class names for only the classes that appear in the test dataset
        test_class_names = [class_names[i] for i in unique_classes]

        # Generate classification report with only the classes that appear in the test dataset
        try:
            report = classification_report(targets, predictions,
                                          labels=unique_classes,
                                          target_names=test_class_names,
                                          output_dict=True)
            print(f"Generated classification report for {len(unique_classes)} classes")
        except Exception as e:
            print(f"Error generating classification report: {e}")
            # Create a minimal report
            report = {
                'accuracy': accuracy,
                'weighted avg': {
                    'precision': precision,
                    'recall': recall,
                    'f1-score': f1,
                    'support': len(targets)
                }
            }
    except Exception as e:
        print(f"Error during testing: {e}")
        print(f"Please check that the test directory '{test_dir}' exists and contains valid data.")
        return None

    # Print results
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Save results to CSV
    results_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Value': [accuracy, precision, recall, f1]
    })
    results_df.to_csv(os.path.join(output_dir, 'test_results.csv'), index=False)

    # Save detailed classification report
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_dir, 'classification_report.csv'))

    # Plot confusion matrix
    print("Generating confusion matrix...")
    plot_confusion_matrix(targets, predictions, class_names, output_dir)

    # Plot sample reconstructions
    print("Generating sample reconstructions...")
    plot_reconstruction_samples(model, test_loader, device, output_dir)

    # Plot latent space visualization
    print("Generating latent space visualization...")
    plot_latent_space(model, test_loader, device, output_dir)

    # Save predictions
    predictions_df = pd.DataFrame({
        'True': [idx_to_class[i] for i in targets],
        'Predicted': [idx_to_class[i] for i in predictions]
    })
    predictions_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)

    print(f"Test results saved to {output_dir}")

    # Return results
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'report': report,
        'predictions': predictions,
        'targets': targets,
        'class_probs': class_probs
    }

if __name__ == "__main__":
    test_combined_model()
