# Hybrid VAE-ViT Model for Plant Disease Classification

## Overview

This project implements a hybrid deep learning architecture that combines Variational Autoencoders (VAE) and Vision Transformers (ViT) for plant disease classification. The model leverages the VAE's ability to learn compact latent representations of plant leaf images and the ViT's capability to model long-range dependencies through self-attention mechanisms.

Plant diseases cause significant economic losses in agriculture worldwide. Early and accurate detection is crucial for effective disease management. This hybrid approach aims to provide an efficient and accurate solution for automated plant disease detection.

## Installation

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- CUDA-compatible GPU (recommended for training)
- CUDA and cuDNN installed (for GPU acceleration)
- Minimum 8GB RAM (16GB+ recommended for training)
- Storage space for the dataset (~2GB)

### Environment Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd vit+vae
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   # Using conda
   conda create -n vae-vit python=3.8
   conda activate vae-vit

   # OR using venv
   python -m venv vae-vit-env
   # On Windows
   vae-vit-env\Scripts\activate
   # On Linux/Mac
   source vae-vit-env/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Verify GPU availability (optional):
   ```bash
   python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"
   ```

5. Configure data paths:
   - Open `config/model_config.py`
   - Update the `TRAIN_DIR`, `VALID_DIR`, and `TEST_DIR` paths to point to your dataset

## Project Structure

```
vit+vae/
├── config/                  # Configuration files
│   └── model_config.py      # Model and training parameters
├── data/                    # Data handling code
│   └── data_loader.py       # Data loading and preprocessing
├── models/                  # Model definitions
│   ├── vae.py               # VAE model
│   ├── vit.py               # Vision Transformer model
│   └── hybrid.py            # Hybrid model combining VAE and ViT
├── training/                # Training scripts
│   ├── train_vae.py         # VAE training script
│   └── train_hybrid.py      # Hybrid model training script
├── inference/               # Inference scripts
│   ├── inference.py         # Single image inference
│   └── batch_inference.py   # Batch inference for multiple images
├── utils/                   # Utility functions
│   ├── model_utils.py       # Model loading utilities
│   └── visualization.py     # Visualization utilities
├── saved_models/            # Saved model weights
│   ├── vae/                 # VAE model weights
│   └── hybrid/              # Hybrid model weights
├── checkpoints/             # Training checkpoints
├── inference_results/       # Results from inference
├── main.py                  # Main entry point
├── requirements.txt         # Project dependencies
└── README.md                # This file
```

## How to Run the Project

The project uses a command-line interface through `main.py` for all operations. Before running any commands, make sure you have configured the data paths in `config/model_config.py`.

### Complete Training Pipeline

The training process consists of two sequential steps:

1. First, train the VAE to learn a good latent representation
2. Then, train the hybrid model using the pre-trained VAE encoder

### Step 1: Training the VAE

```bash
# Navigate to the project directory
cd vit+vae

# Train the VAE for 10 epochs (adjust as needed)
python main.py train-vae --epochs 10
```

During VAE training:
- The model will learn to encode and decode plant leaf images
- Checkpoints will be saved in `checkpoints/vae/`
- Final weights will be saved in `saved_models/vae/`
- Progress will be displayed with loss metrics

### Step 2: Training the Hybrid Model

After the VAE training is complete, train the hybrid model:

```bash
# Train the hybrid model for 10 epochs (adjust as needed)
python main.py train-hybrid --epochs 10
```

During hybrid model training:
- The pre-trained VAE encoder weights will be loaded and frozen
- Only the Vision Transformer component will be trained
- Checkpoints will be saved in `checkpoints/hybrid/`
- Final weights will be saved in `saved_models/hybrid/`
- Training metrics (accuracy, loss) will be displayed

### Testing and Evaluation

#### Single Image Inference

To classify a single plant leaf image:

```bash
# Basic inference
python main.py inference --image path/to/your/image.jpg

# Specify a custom directory to save results
python main.py inference --image path/to/your/image.jpg --save_dir custom_results
```

This will:
1. Load the trained hybrid model
2. Process the specified image
3. Predict the plant disease class with confidence score
4. Display and save a visualization with the prediction
5. Print inference time and other metrics

#### Batch Testing on Multiple Images

To run inference on a directory of images:

```bash
# Basic batch inference
python main.py batch --input_dir path/to/images --output_dir batch_results

# With additional options
python main.py batch --input_dir path/to/images --output_dir batch_results --batch_size 32 --visualize
```

This will:
1. Process all images in the input directory
2. Generate predictions for each image
3. Save results to a CSV file in the output directory
4. Calculate and display average inference time
5. Optionally generate visualizations if `--visualize` is specified

### Example Workflow

Here's a complete example workflow:

```bash
# 1. Configure data paths in config/model_config.py

# 2. Train the VAE
python main.py train-vae --epochs 20

# 3. Train the hybrid model
python main.py train-hybrid --epochs 30

# 4. Run inference on a test image
python main.py inference --image C:/Users/Prudvi/Downloads/archive/test/test/Apple___Apple_scab/0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG

# 5. Run batch inference on a test directory
python main.py batch --input_dir C:/Users/Prudvi/Downloads/archive/test/test --output_dir test_results --visualize
```

### Monitoring and Visualization

- Training progress is displayed in the console
- Model checkpoints are saved at regular intervals
- Inference results include visualizations of the predictions
- Batch inference results are saved in CSV format for further analysis

## Model Architecture

The hybrid architecture consists of two main components:

### 1. Variational Autoencoder (VAE)
- **Encoder**: Compresses 224×224×3 RGB images into a 256-dimensional latent space
- **Decoder**: Reconstructs images from the latent space (used only during VAE training)

### 2. Vision Transformer (ViT)
- Takes the 256-dimensional latent vector from the VAE encoder
- Processes it through transformer blocks with self-attention mechanisms
- Outputs a probability distribution over 38 plant disease classes

## Dataset and Data Preparation

### Dataset Information

The model is trained on the "New Plant Diseases Dataset," which contains images of healthy and diseased plant leaves across various crops:
- 38 classes (different plant diseases and healthy plants)
- Images are resized to 224×224 pixels
- Data augmentation is applied during training

### Data Preparation

1. **Download the Dataset**:
   - The dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
   - Extract the downloaded archive to a location on your computer

2. **Configure Data Paths**:
   - Open `config/model_config.py`
   - Update the following paths to point to your dataset:
     ```python
     TRAIN_DIR = "path/to/New Plant Diseases Dataset(Augmented)/train"
     VALID_DIR = "path/to/New Plant Diseases Dataset(Augmented)/valid"
     TEST_DIR = "path/to/test/test"
     ```

3. **Data Structure**:
   The dataset should have the following structure:
   ```
   train/
   ├── Apple___Apple_scab/
   ├── Apple___Black_rot/
   ├── Apple___Cedar_apple_rust/
   └── ... (other classes)

   valid/
   ├── Apple___Apple_scab/
   ├── Apple___Black_rot/
   ├── Apple___Cedar_apple_rust/
   └── ... (other classes)

   test/
   ├── Apple___Apple_scab/
   ├── Apple___Black_rot/
   ├── Apple___Cedar_apple_rust/
   └── ... (other classes)
   ```

4. **Data Preprocessing**:
   - Images are automatically resized to 224×224 pixels during loading
   - Pixel values are normalized to the range [0, 1]
   - Data augmentation (horizontal flips, rotation, zoom) is applied during training

## Performance

The hybrid VAE-ViT model achieves competitive accuracy on plant disease classification tasks while being computationally efficient due to the dimensionality reduction provided by the VAE.

## Future Improvements

- Deeper VAE architecture for better feature extraction
- More transformer layers for enhanced pattern recognition
- Attention visualization for improved interpretability
- Gradual unfreezing of the VAE encoder during training
- Ensemble approaches for increased robustness

## Troubleshooting

### Common Issues and Solutions

#### Model Loading Issues
- **Problem**: `Could not load VAE encoder weights` or `Could not find model weights in any of the expected locations`
- **Solution**:
  - Ensure you've completed the VAE training step before training the hybrid model
  - Check that all model weights are in the correct directories:
    - VAE weights should be in `saved_models/vae/`
    - Hybrid model weights should be in `saved_models/hybrid/`
  - If weights are missing, run the training process again

#### GPU/Memory Issues
- **Problem**: `CUDA out of memory` or training is very slow
- **Solution**:
  - Reduce batch size in `config/model_config.py` (try 16 or 8)
  - Close other GPU-intensive applications
  - If using CPU only, expect significantly longer training times
  - Consider using Google Colab or another cloud service with GPU access

#### Data Loading Issues
- **Problem**: `No such file or directory` or `Found 0 images belonging to 0 classes`
- **Solution**:
  - Double-check the data paths in `config/model_config.py`
  - Ensure the dataset is downloaded and extracted correctly
  - Verify the dataset directory structure matches the expected format

#### Dependency Issues
- **Problem**: `ModuleNotFoundError` or `ImportError`
- **Solution**:
  - Ensure all packages in `requirements.txt` are installed:
    ```bash
    pip install -r requirements.txt
    ```
  - Check for version conflicts between packages
  - Try creating a fresh virtual environment

#### Inference Issues
- **Problem**: Poor prediction results or unexpected outputs
- **Solution**:
  - Verify that both VAE and hybrid model were trained properly
  - Check that the input image format matches the expected format (224×224 RGB)
  - Try running inference on known test images from the dataset first
  - Increase the number of training epochs if accuracy is low

## License

[Specify your license here]

## Acknowledgments

- The New Plant Diseases Dataset creators
- TensorFlow and Keras development teams
