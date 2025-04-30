# Hybrid VAE-ViT Model for Plant Disease Classification (PyTorch Implementation)

This is a PyTorch implementation of the hybrid Variational Autoencoder (VAE) and Vision Transformer (ViT) model for plant disease classification.

## Overview

This project implements a hybrid deep learning architecture that combines Variational Autoencoders (VAE) and Vision Transformers (ViT) for plant disease classification. The model leverages the VAE's ability to learn compact latent representations of plant leaf images and the ViT's capability to model long-range dependencies through self-attention mechanisms.

## Project Structure

```
vit+vae-pytorch/
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
│   └── batch_inference.py   # Batch inference
├── utils/                   # Utility functions
│   └── model_utils.py       # Model loading and management
├── main.py                  # Main entry point
└── simple_train_vae.py      # Simplified VAE training script
```

## Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.7+
- torchvision
- matplotlib
- pandas
- tqdm
- Pillow

### Setup

1. Clone the repository
2. Install the required packages:
   ```bash
   pip install torch torchvision matplotlib pandas tqdm Pillow
   ```
3. Configure the data paths in `config/model_config.py`

## Usage

The project uses a command-line interface through `main.py` for all operations.

### Complete Training Pipeline

The training process consists of two sequential steps:

1. First, train the VAE to learn a good latent representation
2. Then, train the hybrid model using the pre-trained VAE encoder

### Step 1: Training the VAE

```bash
# Navigate to the project directory
cd vit+vae-pytorch

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

### Inference

For single image inference:

```bash
python main.py inference --image path/to/image.jpg --visualize
```

For batch inference:

```bash
python main.py batch --input_dir path/to/images --output_dir batch_results --batch_size 32 --visualize
```

## Model Architecture

### 1. Variational Autoencoder (VAE)
- **Encoder**: Compresses 224×224×3 RGB images into a 256-dimensional latent space
- **Decoder**: Reconstructs images from the latent space (used only during VAE training)

### 2. Vision Transformer (ViT)
- Takes the 256-dimensional latent vector from the VAE encoder
- Processes it through transformer blocks with self-attention mechanisms
- Outputs a probability distribution over 38 plant disease classes

## Dataset

The model is designed to work with the Plant Village dataset, which contains images of plant leaves with various diseases. The dataset should be organized in the following structure:

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
```

## Troubleshooting

### Common Issues and Solutions

#### GPU/Memory Issues
- **Problem**: `CUDA out of memory` or training is very slow
- **Solution**:
  - Reduce batch size in `config/model_config.py` (try 16 or 8)
  - Close other GPU-intensive applications
  - If using CPU only, expect significantly longer training times

#### Data Loading Issues
- **Problem**: `No such file or directory` or `Found 0 images belonging to 0 classes`
- **Solution**:
  - Double-check the data paths in `config/model_config.py`
  - Ensure the dataset is downloaded and extracted correctly
  - Verify the dataset directory structure matches the expected format

## License

This project is licensed under the MIT License - see the LICENSE file for details.
