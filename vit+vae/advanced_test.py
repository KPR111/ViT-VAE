import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
from load_hybrid import load_trained_hybrid_model
from loadingpreprocessing import train_generator  # To get class indices

# Define the test data path
TEST_DIR = r"C:\Users\Prudvi\Downloads\archive\test\test"

def extract_class_from_filename(filename):
    """Extract class name from filename by removing digits and extension"""
    # Remove digits and file extension
    class_name = ''.join([i for i in filename if not i.isdigit()]).split('.')[0]
    return class_name

def map_to_standard_class(extracted_class, class_indices):
    """Map the extracted class name to the standard class name in the model"""
    # This is a simple mapping - you may need to adjust based on your class names
    for standard_class in class_indices.keys():
        if extracted_class.lower() in standard_class.lower() or standard_class.lower() in extracted_class.lower():
            return standard_class
    return None  # No match found

def visualize_predictions(images, true_classes, pred_classes, confidences, class_indices, save_dir):
    """Visualize and save prediction results"""
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
    plt.savefig(os.path.join(save_dir, "prediction_grid.png"))
    plt.close()
    
    # Create confusion matrix
    unique_classes = sorted(list(set(true_classes + pred_classes)))
    cm = confusion_matrix(
        [unique_classes.index(c) for c in true_classes],
        [unique_classes.index(c) for c in pred_classes],
        labels=range(len(unique_classes))
    )
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=unique_classes, yticklabels=unique_classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()

def main():
    print("Loading the trained hybrid model...")
    # Load the trained hybrid model
    hybrid_model = load_trained_hybrid_model()
    
    if hybrid_model is None:
        print("Failed to load the hybrid model. Make sure you have trained the model first.")
        return
    
    # Get class names from training data
    class_indices = train_generator.class_indices
    class_names = list(class_indices.keys())
    idx_to_class = {v: k for k, v in class_indices.items()}
    
    print(f"Model has {len(class_names)} classes")
    
    # Load and preprocess test images
    test_images = []
    test_image_paths = []
    original_images = []  # For visualization
    
    # Get all image files from the test directory
    for filename in os.listdir(TEST_DIR):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff')):
            img_path = os.path.join(TEST_DIR, filename)
            test_image_paths.append(img_path)
            
            # Load and preprocess the image for the model
            img = load_img(img_path, target_size=(224, 224))
            img_array = img_to_array(img) / 255.0
            test_images.append(img_array)
            
            # Also keep original image for visualization
            original_images.append(plt.imread(img_path))
    
    # Convert to numpy array
    test_images = np.array(test_images)
    
    print(f"Loaded {len(test_images)} test images")
    
    # Make predictions
    print("Making predictions on test images...")
    predictions = hybrid_model.predict(test_images)
    predicted_classes_idx = np.argmax(predictions, axis=1)
    predicted_classes = [idx_to_class[idx] for idx in predicted_classes_idx]
    confidences = [predictions[i][idx] * 100 for i, idx in enumerate(predicted_classes_idx)]
    
    # Extract ground truth from filenames
    true_classes = []
    for path in test_image_paths:
        filename = os.path.basename(path)
        extracted_class = extract_class_from_filename(filename)
        mapped_class = map_to_standard_class(extracted_class, class_indices)
        
        if mapped_class:
            true_classes.append(mapped_class)
        else:
            # If no mapping found, use the extracted class
            print(f"Warning: Could not map {extracted_class} to a standard class")
            true_classes.append(extracted_class)
    
    # Create results directory
    results_dir = "test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save detailed results to a text file
    with open(os.path.join(results_dir, "detailed_predictions.txt"), "w") as f:
        f.write("Image\tTrue Class\tPredicted Class\tConfidence\tCorrect?\n")
        f.write("-" * 80 + "\n")
        
        for i, (image_path, true_class, pred_class, confidence) in enumerate(
            zip(test_image_paths, true_classes, predicted_classes, confidences)):
            
            image_name = os.path.basename(image_path)
            is_correct = true_class == pred_class
            result_line = f"{image_name}\t{true_class}\t{pred_class}\t{confidence:.2f}%\t{is_correct}"
            f.write(result_line + "\n")
    
    # Calculate accuracy (if we have true classes)
    if all(c in class_names for c in true_classes):
        accuracy = accuracy_score([class_indices.get(c, -1) for c in true_classes], 
                                 predicted_classes_idx)
        print(f"\nTest Accuracy: {accuracy:.4f}")
        
        # Save classification report
        report = classification_report(true_classes, predicted_classes)
        print("\nClassification Report:")
        print(report)
        
        with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
            f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
    
    # Visualize predictions
    visualize_predictions(original_images, true_classes, predicted_classes, 
                         confidences, class_indices, results_dir)
    
    print(f"\nResults saved to '{results_dir}' directory")
    print("Done!")

if __name__ == "__main__":
    main()
