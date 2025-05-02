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
│   ├── train_combined.py    # Combined model training script
│   └── test_combined.py     # Testing script for evaluation
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

### Testing

After training the model, you can evaluate its performance on a test dataset:

```bash
python main.py test --test_dir path/to/test/dataset --output_dir test_results
```

The test directory can have a flat structure with all images in a single directory, where the class name is part of the filename (e.g., "AppleScab3.JPG"). The testing script will automatically extract the class name from the filename. You can specify a custom test directory or use the default one configured in `config/model_config.py`.

#### Checking Test Directory

If you're having issues with the test command, you can check your test directory structure:

```bash
python main.py check-test-dir --test_dir path/to/test/dataset
```

This will:
- List all files in the test directory
- Count the number of image files
- Show sample image filenames
- Display the extracted class names from filenames

#### Testing Results

The test command will:
- Evaluate the model on the test dataset
- Calculate accuracy, precision, recall, and F1 score
- Generate a confusion matrix
- Show sample reconstructions from the VAE component
- Visualize the latent space
- Save all results to the specified output directory

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

Our model combines a Variational Autoencoder (VAE) and a Vision Transformer (ViT) in a unified architecture that is trained end-to-end with a joint loss function. This approach allows the VAE to learn latent representations that are both good for reconstruction and optimized for classification.

#### Simultaneous Training Approach

Unlike traditional methods that might pre-train the VAE separately and then train a classifier on its frozen latent space, our approach:

1. **Trains both components simultaneously** from scratch
2. **Uses a combined loss function** that incorporates both reconstruction and classification objectives
3. **Allows gradient flow** between components, enabling the VAE to learn representations that directly benefit classification
4. **Applies KL annealing** to balance reconstruction quality and latent space regularization

This simultaneous training creates a more cohesive model where the latent space serves both reconstruction and classification purposes effectively.

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
- **Total loss** = Reconstruction loss + Weighted KL divergence loss + Classification loss
- **Weighting strategy**: KL weight is gradually increased during training (annealing)
- **Benefits**: Prevents posterior collapse while ensuring both good reconstruction and classification performance

## Detailed Architecture and Dimension Flow

### Variational Autoencoder (VAE) Architecture in Detail

The VAE consists of an encoder that compresses images into a latent space and a decoder that reconstructs images from this latent representation. The probabilistic nature of the latent space enables both effective compression and generative capabilities.

#### Configuration Parameters
- **IMAGE_SIZE** = (224, 224): Input image dimensions
- **IMAGE_CHANNELS** = 3: RGB color channels
- **LATENT_DIM** = 256: Dimensionality of the latent space
- **BATCH_SIZE** = 32: Number of samples processed in each batch
- **KL_WEIGHT** = 0.01: Maximum weight for the KL divergence term
- **KL_ANNEALING_EPOCHS** = 10: Number of epochs to gradually increase KL weight

#### Encoder Architecture

| Stage | Operation | Input Shape | Output Shape | Parameters |
|-------|-----------|-------------|--------------|------------|
| **Input** | RGB Image | [B, C, H, W] = [32, 3, 224, 224] | [32, 3, 224, 224] | - |
| **Conv1** | Conv2d + BN + ReLU | [32, 3, 224, 224] | [32, 32, 112, 112] | stride=2, kernel=3, padding=1 |
| **Conv2** | Conv2d + BN + ReLU | [32, 32, 112, 112] | [32, 64, 56, 56] | stride=2, kernel=3, padding=1 |
| **Conv3** | Conv2d + BN + ReLU | [32, 64, 56, 56] | [32, 128, 28, 28] | stride=2, kernel=3, padding=1 |
| **Flatten** | View | [32, 128, 28, 28] | [32, 100352] | 128 × 28 × 28 = 100,352 |
| **FC** | Linear + BN + ReLU | [32, 100352] | [32, 256] | - |
| **Mean** | Linear | [32, 256] | [32, 256] | Projects to mean vector |
| **LogVar** | Linear | [32, 256] | [32, 256] | Projects to log variance |
| **Sampling** | Reparameterization | [32, 256], [32, 256] | [32, 256] | z = mean + std × epsilon |

#### Decoder Architecture

| Stage | Operation | Input Shape | Output Shape | Parameters |
|-------|-----------|-------------|--------------|------------|
| **Input** | Latent Vector | [B, L] = [32, 256] | [32, 256] | - |
| **FC** | Linear + BN + ReLU | [32, 256] | [32, 25088] | 256 → 28 × 28 × 32 = 25,088 |
| **Reshape** | View | [32, 25088] | [32, 32, 28, 28] | Reshape to feature maps |
| **Deconv1** | ConvTranspose2d + BN + ReLU | [32, 32, 28, 28] | [32, 64, 56, 56] | stride=2, kernel=3, padding=1, output_padding=1 |
| **Deconv2** | ConvTranspose2d + BN + ReLU | [32, 64, 56, 56] | [32, 32, 112, 112] | stride=2, kernel=3, padding=1, output_padding=1 |
| **Deconv3** | ConvTranspose2d + BN + ReLU | [32, 32, 112, 112] | [32, 16, 224, 224] | stride=2, kernel=3, padding=1, output_padding=1 |
| **DeconvFinal** | ConvTranspose2d + Sigmoid | [32, 16, 224, 224] | [32, 3, 224, 224] | kernel=3, padding=1 |

#### Visual Representation of VAE Architecture

```
                                 VAE ARCHITECTURE
                                 ----------------
                                        ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                               ENCODER                                    │
│                                                                         │
│  Input Image [32, 3, 224, 224]                                          │
│         ↓                                                               │
│  Conv1 + BN + ReLU [32, 32, 112, 112]                                   │
│         ↓                                                               │
│  Conv2 + BN + ReLU [32, 64, 56, 56]                                     │
│         ↓                                                               │
│  Conv3 + BN + ReLU [32, 128, 28, 28]                                    │
│         ↓                                                               │
│  Flatten [32, 100352]                                                   │
│         ↓                                                               │
│  FC + BN + ReLU [32, 256]                                               │
│         ↓                                                               │
│  ┌─────────────────┐          ┌─────────────────┐                       │
│  │ Mean [32, 256]  │          │ LogVar [32, 256]│                       │
│  └─────────────────┘          └─────────────────┘                       │
│         ↓                            ↓                                  │
│         └─────────→ Sampling ←───────┘                                  │
│                       ↓                                                 │
│                Latent Vector z [32, 256]                                │
└─────────────────────────────┬───────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                               DECODER                                    │
│                                                                         │
│  Latent Vector z [32, 256]                                              │
│         ↓                                                               │
│  FC + BN + ReLU [32, 25088]                                             │
│         ↓                                                               │
│  Reshape [32, 32, 28, 28]                                               │
│         ↓                                                               │
│  Deconv1 + BN + ReLU [32, 64, 56, 56]                                   │
│         ↓                                                               │
│  Deconv2 + BN + ReLU [32, 32, 112, 112]                                 │
│         ↓                                                               │
│  Deconv3 + BN + ReLU [32, 16, 224, 224]                                 │
│         ↓                                                               │
│  DeconvFinal + Sigmoid [32, 3, 224, 224]                                │
│         ↓                                                               │
│  Reconstructed Image [32, 3, 224, 224]                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Reparameterization Trick

The VAE uses the reparameterization trick to enable backpropagation through the sampling process:

1. **Encoder outputs**: mean (μ) and log variance (log σ²) vectors
2. **Standard deviation**: σ = exp(0.5 × log σ²)
3. **Random noise**: ε ~ N(0, 1) (standard normal distribution)
4. **Sampling**: z = μ + σ × ε

This approach allows gradient flow through the sampling operation during training.

#### Key Components of the VAE

1. **Convolutional Layers**: Extract hierarchical features from images with progressively reduced spatial dimensions and increased channel depth
2. **Batch Normalization**: Stabilizes and accelerates training by normalizing activations
3. **Latent Space**: Probabilistic representation where each point is defined by a mean and variance
4. **Transposed Convolutions**: Upsample the latent representation back to image dimensions
5. **Skip Connections**: Not used in this implementation, but could be added to preserve fine details
6. **Weight Initialization**: Xavier/Glorot initialization for stable training

### Vision Transformer (ViT) Architecture in Detail

The Vision Transformer component processes the latent vector from the VAE encoder and transforms it into class predictions. Unlike traditional ViTs that operate directly on image patches, our model works with the compressed latent representation.

#### Configuration Parameters
- **LATENT_DIM (L)** = 256: Size of the input latent vector
- **BATCH_SIZE (B)** = 32: Number of samples processed in each batch
- **PATCH_SIZE (P)** = 16: Size of each patch's feature vector
- **NUM_PATCHES (N)** = 16: Number of patches (LATENT_DIM ÷ PATCH_SIZE = 256 ÷ 16 = 16)
- **NUM_CLASSES (C)** = 38: Number of output classes (plant disease categories)
- **NUM_TRANSFORMER_LAYERS** = 6: Number of transformer blocks
- **NUM_HEADS** = 8: Number of attention heads
- **DROPOUT_RATE** = 0.1: Dropout probability

#### Dimension Flow Through the ViT

| Stage | Operation | Input Shape | Output Shape | Description |
|-------|-----------|-------------|--------------|-------------|
| **Input** | Latent Vector | [B, L] = [32, 256] | [32, 256] | Latent representation from VAE encoder |
| **Dense Layer** | Linear(256, 256) | [32, 256] | [32, 256] | Transforms latent vector while preserving dimensions |
| **Reshape** | view(B, N, P) | [32, 256] | [32, 16, 16] | Reshapes to sequence of patches for transformer processing |
| **Transformer Blocks** | 6 layers of self-attention | [32, 16, 16] | [32, 16, 16] | Each block processes relationships between patches |
| **Layer Norm** | LayerNorm(16) | [32, 16, 16] | [32, 16, 16] | Normalizes features for stable processing |
| **Flatten** | Flatten() | [32, 16, 16] | [32, 256] | Prepares for classification |
| **Dropout** | Dropout(0.3) | [32, 256] | [32, 256] | Prevents overfitting |
| **Classification** | Linear(256, 38) | [32, 256] | [32, 38] | Maps to class logits |
| **Softmax** | Softmax(dim=1) | [32, 38] | [32, 38] | Converts to class probabilities |

#### Transformer Block Details

Each transformer block consists of:

1. **Layer Normalization 1**: Normalizes input features
2. **Multi-Head Attention**:
   - Input: [32, 16, 16]
   - 8 attention heads, each processing 16 ÷ 8 = 2 dimensions
   - Self-attention across the 16 patches
   - Output: [32, 16, 16]
3. **Skip Connection**: Adds input to attention output
4. **Layer Normalization 2**: Normalizes features again
5. **MLP Block**:
   - Linear layer: [16 → 16]
   - GELU activation
   - Dropout (0.1)
   - Output: [32, 16, 16]
6. **Skip Connection**: Adds normalized input to MLP output

#### Visual Representation of Dimension Flow

```
Input Latent Vector [32, 256]
       ↓
Dense Layer [32, 256]
       ↓
Reshape to Patches [32, 16, 16]
       ↓
┌─────────────────────────┐
│ Transformer Block 1     │
│  ┌─────┐  ┌───────────┐ │
│  │Norm1│→ │Self-Attn  │ │
│  └─────┘  └───────────┘ │
│      ↓        ↓         │
│      └────→ (+)         │
│             ↓           │
│  ┌─────┐  ┌───────────┐ │
│  │Norm2│→ │MLP Block  │ │
│  └─────┘  └───────────┘ │
│      ↓        ↓         │
│      └────→ (+)         │
└─────────────↓───────────┘
       ↓
     ... (5 more blocks)
       ↓
Layer Normalization [32, 16, 16]
       ↓
Flatten [32, 256]
       ↓
Classification Head [32, 38]
       ↓
Output Probabilities [32, 38]
```

#### Key Insights

- **Patch Processing**: Instead of processing image patches directly, our ViT processes "patches" of the latent vector
- **Attention Mechanism**: Each attention head focuses on different aspects of relationships between latent patches
- **Dimensionality**: The model maintains the total information (256 dimensions) throughout processing until the final classification layer
- **Efficiency**: Working with the compressed latent representation (256D) rather than raw images (224×224×3) significantly reduces computational requirements

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
