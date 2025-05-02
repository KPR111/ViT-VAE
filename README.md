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
│   └── combined_model.py    # Combined model (VAE and ViT trained simultaneously)
├── training/                # Training scripts
│   └── train_combined.py    # Combined model training script
├── inference/               # Inference scripts
│   ├── inference.py         # Single image inference
│   └── batch_inference.py   # Batch inference
├── utils/                   # Utility functions
│   └── model_utils.py       # Model loading and management
└── main.py                  # Main entry point
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

### Training the Model

Train the combined model where both VAE and ViT components are trained simultaneously:

```bash
# Train the combined model for 10 epochs (adjust as needed)
python main.py train-combined --epochs 10
```

During combined model training:
- Both the VAE and ViT components are trained simultaneously
- The model uses a combined loss function (VAE reconstruction + KL divergence + classification)
- Checkpoints will be saved in `checkpoints/hybrid/`
- Final weights will be saved in `saved_models/hybrid/`
- Training metrics (reconstruction loss, classification loss, accuracy) will be displayed

### Inference

For single image inference:

```bash
python main.py inference --image path/to/image.jpg --visualize --model combined
```

For batch inference:

```bash
python main.py batch --input_dir path/to/images --output_dir batch_results --batch_size 32 --visualize --model combined
```

## Model Architecture

### Combined VAE-ViT Model
The combined model trains both components simultaneously with a joint loss function:

#### 1. Variational Autoencoder (VAE)
- **Encoder**: Compresses 224×224×3 RGB images into a 256-dimensional latent space
- **Decoder**: Reconstructs images from the latent space
- **Loss**: Reconstruction loss (MSE) + KL divergence loss with annealing

#### 2. Vision Transformer (ViT)
- Takes the 256-dimensional latent vector from the VAE encoder
- Processes it through transformer blocks with self-attention mechanisms
- Outputs a probability distribution over plant disease classes
- **Loss**: Cross-entropy classification loss

#### 3. Combined Loss
- Total loss = Reconstruction loss + Weighted KL divergence loss + Classification loss
- KL weight is gradually increased during training (annealing)

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
