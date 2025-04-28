# Hybrid VAE-ViT Model for Plant Disease Classification

## Problem Statement

Plant diseases cause significant economic losses in agriculture worldwide. Early and accurate detection of plant diseases is crucial for effective disease management and to minimize crop losses. Traditional methods of plant disease detection rely on visual inspection by experts, which is time-consuming, subjective, and often requires specialized knowledge.

This project addresses the challenge of automated plant disease detection using a novel hybrid deep learning approach that combines the strengths of Variational Autoencoders (VAE) and Vision Transformers (ViT).

## Project Overview

This project implements a hybrid deep learning architecture that combines:

1. **Variational Autoencoder (VAE)**: For efficient feature extraction and dimensionality reduction
2. **Vision Transformer (ViT)**: For powerful classification capabilities using attention mechanisms

The hybrid model leverages the VAE's ability to learn a compact latent representation of plant leaf images and the ViT's capability to model long-range dependencies in the data through self-attention mechanisms.

## Why This Approach?

### Advantages of the Hybrid VAE-ViT Architecture

1. **Efficient Feature Extraction**: The VAE compresses the high-dimensional image data into a lower-dimensional latent space, capturing the most important features while reducing computational requirements.

2. **Robust to Variations**: The VAE's probabilistic nature helps the model generalize better across different lighting conditions, angles, and other variations in leaf images.

3. **Attention-Based Classification**: The Vision Transformer excels at capturing global dependencies in the data, allowing it to focus on the most relevant parts of the latent representation for disease classification.

4. **Transfer Learning Potential**: The modular architecture allows for pre-training components separately and fine-tuning the complete model, enhancing performance with limited data.

5. **Interpretability**: The attention mechanisms in the ViT component can potentially highlight which parts of the latent representation are most important for classification decisions.

## Architecture Details

### Model Components

#### 1. VAE Encoder
- Input: RGB images of size 224×224×3
- Architecture:
  - Conv2D layer: 32 filters, 3×3 kernel, stride 2, 'same' padding, ReLU activation
  - Conv2D layer: 64 filters, 3×3 kernel, stride 2, 'same' padding, ReLU activation
  - Flatten layer
  - Dense layer: 128 units, ReLU activation
  - Two parallel Dense layers for z_mean and z_log_var (256 units each)
  - Sampling layer using the reparameterization trick
- Output: 256-dimensional latent vector

#### 2. VAE Decoder (used only during VAE training)
- Input: 256-dimensional latent vector
- Architecture:
  - Dense layer: 56×56×32 units, ReLU activation
  - Reshape to 56×56×32
  - Conv2DTranspose layer: 32 filters, 3×3 kernel, stride 2, 'same' padding, ReLU activation
  - Conv2DTranspose layer: 16 filters, 3×3 kernel, stride 2, 'same' padding, ReLU activation
  - Conv2DTranspose layer: 3 filters, 3×3 kernel, 'same' padding, sigmoid activation
- Output: Reconstructed RGB image of size 224×224×3

#### 3. Vision Transformer (ViT)
- Input: 256-dimensional latent vector from VAE encoder
- Architecture:
  - Dense expansion layer: 256 units
  - Reshape to 16 patches of size 16 (16×16=256)
  - 4 Transformer encoder blocks, each containing:
    - Layer normalization
    - Multi-head self-attention (4 heads, key dimension 16)
    - Skip connection
    - Layer normalization
    - MLP block with GELU activation
    - Skip connection
  - Final layer normalization
  - Flatten layer
  - Dropout (0.3)
  - Dense layer: 38 units (number of classes), softmax activation
- Output: Probability distribution over 38 plant disease classes

#### 4. Hybrid Model
- The VAE encoder and ViT are combined into a single end-to-end model
- During training, the VAE encoder weights are frozen (transfer learning)
- Only the ViT component is trained on the plant disease classification task

## Dimension Changes Throughout the Architecture

### VAE Encoder
1. Input: 224×224×3 = 150,528 dimensions
2. First Conv2D: 112×112×32 = 401,408 dimensions
3. Second Conv2D: 56×56×64 = 200,704 dimensions
4. Flatten: 200,704 dimensions
5. Dense: 128 dimensions
6. z_mean and z_log_var: 256 dimensions each
7. Latent vector z: 256 dimensions

### Vision Transformer
1. Input: 256-dimensional latent vector
2. Dense expansion: 256 dimensions
3. Reshape to patches: 16 patches × 16 dimensions
4. Transformer blocks: Maintain 16×16 dimensions through self-attention and MLP operations
5. Flatten: 256 dimensions
6. Output layer: 38 dimensions (class probabilities)

## Learnable Parameters Count

### VAE Encoder
- First Conv2D: (3×3×3×32) + 32 = 896 parameters
- Second Conv2D: (3×3×32×64) + 64 = 18,496 parameters
- Dense: (200,704×128) + 128 = 25,690,240 parameters
- z_mean: (128×256) + 256 = 33,024 parameters
- z_log_var: (128×256) + 256 = 33,024 parameters
- Total VAE Encoder: ~25.78 million parameters

### Vision Transformer
- Dense expansion: (256×256) + 256 = 65,792 parameters
- Each Transformer block:
  - Multi-head attention: ~16,640 parameters
  - MLP: ~4,368 parameters
  - Layer norms: ~64 parameters
  - Total per block: ~21,072 parameters
- 4 Transformer blocks: 4 × 21,072 = 84,288 parameters
- Final layer norm: 512 parameters
- Output layer: (256×38) + 38 = 9,766 parameters
- Total ViT: ~160,358 parameters

### Total Hybrid Model
- VAE Encoder: ~25.78 million parameters (frozen during hybrid training)
- Vision Transformer: ~160,358 parameters (trainable)
- Total: ~25.94 million parameters (only ~160,358 trainable)

## Training Process

The training process is divided into two phases:

### Phase 1: VAE Training
1. The VAE is trained as an autoencoder to reconstruct plant leaf images
2. Loss function combines reconstruction loss (MSE) and KL divergence
3. Adam optimizer with learning rate 1e-4
4. Checkpoints saved during training
5. Final weights saved for encoder and decoder separately

### Phase 2: Hybrid Model Training
1. Pre-trained VAE encoder weights are loaded and frozen
2. ViT classifier is initialized with random weights
3. The hybrid model is trained on the plant disease classification task
4. Loss function: Categorical cross-entropy
5. Adam optimizer with learning rate 1e-4
6. Early stopping based on validation loss
7. Checkpoints saved during training
8. Final hybrid model weights saved

## Dataset

The model is trained on the "New Plant Diseases Dataset," which contains images of healthy and diseased plant leaves across various crops. The dataset includes:
- 38 classes (different plant diseases and healthy plants)
- Images are resized to 224×224 pixels
- Data augmentation applied during training (horizontal flips, rotation, zoom)

## Potential Improvements

Several strategies could be employed to enhance the current architecture:

1. **Deeper VAE**: Adding more convolutional layers to the VAE encoder could improve feature extraction capabilities.

2. **Larger Latent Space**: Increasing the dimensionality of the latent space might preserve more information, potentially improving classification accuracy.

3. **More Transformer Layers**: Adding more transformer blocks could enhance the model's ability to capture complex patterns in the latent representation.

4. **Attention Visualization**: Implementing mechanisms to visualize attention weights could improve interpretability.

5. **Fine-tuning Strategy**: Instead of completely freezing the VAE encoder, implementing a gradual unfreezing strategy during training might improve performance.

6. **Ensemble Approach**: Combining predictions from multiple models (e.g., CNN-based and hybrid models) could enhance robustness.

7. **Self-supervised Pre-training**: Pre-training the VAE on a larger dataset of plant images before fine-tuning on the disease classification task.

8. **Hyperparameter Optimization**: Systematic tuning of hyperparameters like learning rate, batch size, and model architecture.

9. **Knowledge Distillation**: Training a smaller, more efficient model to mimic the behavior of the larger hybrid model.

10. **Explainability Methods**: Incorporating techniques like Grad-CAM to highlight regions of the input image that influenced the classification decision.

## Conclusion

This hybrid VAE-ViT architecture represents a novel approach to plant disease classification that leverages the strengths of both variational autoencoders and vision transformers. By first compressing the image information into a meaningful latent space and then applying transformer-based attention mechanisms, the model can efficiently and effectively classify plant diseases from leaf images.

The modular nature of the architecture allows for flexible experimentation and improvement, making it a promising approach for agricultural applications and potentially other image classification tasks where both feature extraction and attention mechanisms are beneficial.
