# Hybrid VAE-ViT Model for Plant Disease Classification (PyTorch Implementation)

This is a PyTorch implementation of the hybrid Variational Autoencoder (VAE) and Vision Transformer (ViT) model for plant disease classification.

## Overview

This project implements a hybrid deep learning architecture that combines Variational Autoencoders (VAE) and Vision Transformers (ViT) for plant disease classification. The model leverages the VAE's ability to learn compact latent representations of plant leaf images and the ViT's capability to model long-range dependencies through self-attention mechanisms.

Unlike traditional approaches that might train components separately, our implementation trains both the VAE and ViT simultaneously with a combined loss function. This end-to-end training approach allows the VAE to learn latent representations that are optimized for both reconstruction and classification tasks, resulting in a more cohesive and effective model.

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
The combined model trains both components simultaneously with a joint loss function. This approach allows the VAE to learn latent representations that are both good for reconstruction and useful for classification.

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

### Dimension Changes During Training

#### Input to Latent Space (Encoder)
1. **Input Image**: 224×224×3 (RGB image)
2. **First Conv Layer**: 112×112×32 (stride=2, kernel=3, padding=1)
3. **Second Conv Layer**: 56×56×64 (stride=2, kernel=3, padding=1)
4. **Third Conv Layer**: 28×28×128 (stride=2, kernel=3, padding=1)
5. **Flatten**: 100,352 (28×28×128)
6. **Fully Connected**: 256
7. **Latent Space**: 256 (mean) and 256 (log_var) → 256-dimensional latent vector (z)

#### Latent Space to Reconstruction (Decoder)
1. **Latent Vector**: 256
2. **Fully Connected**: 28×28×32 = 25,088
3. **Reshape**: 28×28×32
4. **First Deconv Layer**: 56×56×64 (stride=2, kernel=3, padding=1, output_padding=1)
5. **Second Deconv Layer**: 112×112×32 (stride=2, kernel=3, padding=1, output_padding=1)
6. **Third Deconv Layer**: 224×224×16 (stride=2, kernel=3, padding=1, output_padding=1)
7. **Final Deconv Layer**: 224×224×3 (kernel=3, padding=1)
8. **Output**: 224×224×3 (reconstructed image)

#### Latent Space to Classification (ViT)
1. **Latent Vector**: 256
2. **Dense Layer**: 256 (expands the latent vector)
3. **Reshape to Patches**: 16 patches × 16 dimensions (PATCH_SIZE=16, NUM_PATCHES=256/16=16)
4. **Transformer Blocks**: 6 layers of self-attention and MLP (maintaining 16×16 dimensions)
5. **Layer Normalization**: 16×16
6. **Flatten**: 256
7. **Classification Head**: 38 (number of plant disease classes)
8. **Output**: 38-dimensional probability distribution

### Understanding KL Annealing and Posterior Collapse

#### KL Annealing
KL annealing is a technique used to gradually increase the weight of the KL divergence term in the VAE loss function during training. In our implementation:

1. **Why it's needed**: Without annealing, the model might prioritize minimizing the KL divergence term early in training, leading to a suboptimal latent space.
2. **Implementation**: The KL weight starts at 0 and linearly increases to its maximum value (0.01) over a specified number of epochs (10 by default).
3. **Formula**: `kl_weight = min(current_epoch / annealing_epochs, 1.0) * kl_weight_max`
4. **Benefits**: Allows the model to first focus on reconstruction, then gradually enforce the latent space distribution constraints.

#### Posterior Collapse
Posterior collapse is a common issue in VAEs where the latent variables are ignored, and the decoder learns to generate outputs independently of the encoder.

1. **The problem**: The KL term encourages the approximate posterior to match the prior, which can lead to the latent variables becoming uninformative.
2. **Prevention in our model**:
   - KL annealing helps prevent posterior collapse by gradually introducing the KL term
   - The combined training objective with classification loss encourages the latent space to retain discriminative information
   - Proper initialization of weights in both encoder and decoder
   - Clipping of log variance to prevent numerical instability

#### Latent Structure in VAEs
The latent space in our VAE is designed to capture meaningful representations of plant leaf images:

1. **Dimensionality**: 256-dimensional space allows for capturing complex features while maintaining computational efficiency
2. **Probabilistic nature**: Each point in latent space is represented by a mean vector and log variance vector
3. **Reparameterization trick**: Enables backpropagation through the sampling process
4. **Joint optimization**: By training with both reconstruction and classification objectives, the latent space learns to capture both visual features (for reconstruction) and discriminative features (for classification)

### Inference Process

During inference, the model processes images through the following steps:

1. **Image Preprocessing**:
   - Resize to 224×224 pixels
   - Convert to RGB format
   - Normalize using ImageNet mean and standard deviation

2. **Encoding**:
   - The preprocessed image is passed through the VAE encoder
   - The encoder produces a 256-dimensional latent vector (z)

3. **Classification**:
   - The latent vector is fed into the Vision Transformer
   - The ViT processes the latent representation through transformer blocks
   - The final layer outputs probabilities for each plant disease class

4. **Visualization** (optional):
   - The VAE decoder can reconstruct the input image from the latent vector
   - This reconstruction can be visualized alongside the original image
   - Comparing the original and reconstructed images provides insight into what features the model captures

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
