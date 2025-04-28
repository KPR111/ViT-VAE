import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from load_hybrid import load_trained_hybrid_model
from loadingpreprocessing import train_generator  # To get class indices

# Define the test data path
TEST_DIR = r"C:\Users\Prudvi\Downloads\archive\test\test"

def main():
    print("Loading the trained hybrid model...")
    # Load the trained hybrid model
    hybrid_model = load_trained_hybrid_model()
    
    if hybrid_model is None:
        print("Failed to load the hybrid model. Make sure you have trained the model first.")
        return
    
    # Create a data generator for test images
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Since test data is not organized in folders by class, we'll process individual images
    test_images = []
    test_image_paths = []
    
    # Get all image files from the test directory
    for filename in os.listdir(TEST_DIR):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff')):
            img_path = os.path.join(TEST_DIR, filename)
            test_image_paths.append(img_path)
            
            # Load and preprocess the image
            img = load_img(img_path, target_size=(224, 224))
            img_array = img_to_array(img) / 255.0
            test_images.append(img_array)
    
    # Convert to numpy array
    test_images = np.array(test_images)
    
    print(f"Loaded {len(test_images)} test images")
    
    # Get class names from training data
    class_names = list(train_generator.class_indices.keys())
    class_indices = {v: k for k, v in train_generator.class_indices.items()}
    
    # Make predictions
    print("Making predictions on test images...")
    predictions = hybrid_model.predict(test_images)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Extract ground truth from filenames (assuming filenames contain class names)
    # This is a simple approach - adjust based on your actual filename format
    true_classes = []
    for path in test_image_paths:
        filename = os.path.basename(path)
        # Extract class name from filename (e.g., "AppleScab1.JPG" -> "AppleScab")
        class_name = ''.join([i for i in filename if not i.isdigit()]).split('.')[0]
        true_classes.append(class_name)
    
    # Display results
    print("\nTest Results:")
    print("-" * 50)
    
    # Create a directory to save results
    os.makedirs("test_results", exist_ok=True)
    
    # Save results to a text file
    with open("test_results/predictions.txt", "w") as f:
        f.write("Image\tPredicted Class\tConfidence\n")
        f.write("-" * 50 + "\n")
        
        for i, (image_path, pred_class_idx, pred) in enumerate(zip(test_image_paths, predicted_classes, predictions)):
            image_name = os.path.basename(image_path)
            pred_class = class_indices[pred_class_idx]
            confidence = pred[pred_class_idx] * 100
            
            result_line = f"{image_name}\t{pred_class}\t{confidence:.2f}%"
            print(result_line)
            f.write(result_line + "\n")
            
            # Plot and save the image with prediction
            plt.figure(figsize=(8, 8))
            img = plt.imread(image_path)
            plt.imshow(img)
            plt.title(f"Predicted: {pred_class} ({confidence:.2f}%)\nFilename: {image_name}")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"test_results/prediction_{i}.png")
            plt.close()
    
    print("\nResults saved to 'test_results' directory")
    print("Done!")

if __name__ == "__main__":
    main()
