# Hybrid VAE-ViT Model for Plant Disease Classification

## Model Architecture Parameters

### VAE Model Parameters
Total parameters: ~32.4 million
- Encoder: ~25.85 million parameters
  - Conv1: 896 parameters
    - Calculation: (kernel_size² × in_channels × out_channels) + out_channels
    - (3 × 3 × 3 × 32) + 32 = 896
  - Conv2: 18,496 parameters
    - Calculation: (3 × 3 × 32 × 64) + 64 = 18,496
  - Conv3: 73,856 parameters
    - Calculation: (3 × 3 × 64 × 128) + 128 = 73,856
  - FC: 25,690,368 parameters
    - Calculation: (128 × 28 × 28 × 256) + 256 = 25,690,368
  - FC_mean: 65,792 parameters
    - Calculation: (256 × 256) + 256 = 65,792
  - FC_logvar: 65,792 parameters
    - Calculation: (256 × 256) + 256 = 65,792

### Vision Transformer (ViT) Parameters
Total parameters: ~5.92 million
- Initial Dense Layer: 65,792 parameters
  - Calculation: (LATENT_DIM × LATENT_DIM) + LATENT_DIM
  - (256 × 256) + 256 = 65,792

- Transformer Blocks (×12): 3,947,520 parameters
  - Per block (328,960 parameters):
    - Multi-head Attention: 262,144 parameters
      - Calculation: (3 × embed_dim × embed_dim) + (embed_dim × embed_dim)
      - (3 × 256 × 256) + (256 × 256) = 262,144
    - Layer Norms (2 per block): 1,024 parameters
      - Calculation: 2 × (2 × embed_dim)
      - 2 × (2 × 256) = 1,024
    - MLP block: 65,792 parameters
      - Calculation: (embed_dim × embed_dim) + embed_dim
      - (256 × 256) + 256 = 65,792
  - Total per transformer block: 328,960
  - Total for 12 blocks: 12 × 328,960 = 3,947,520

- Final Layer Norm: 512 parameters
  - Calculation: 2 × PATCH_SIZE
  - 2 × 256 = 512

- Classification Head: 1,908,774 parameters
  - Calculation: (NUM_PATCHES × PATCH_SIZE × NUM_CLASSES) + NUM_CLASSES
  - (196 × 256 × 38) + 38 = 1,908,774

### Combined Model
Total trainable parameters: ~38.3 million parameters

### Notes on Parameter Calculations:
- For convolutional layers: parameters = (kernel_size² × in_channels × out_channels) + out_channels
- For fully connected layers: parameters = (input_features × output_features) + output_features
- For layer normalization: parameters = 2 × features (gamma and beta parameters)
- For multi-head attention: parameters include query, key, and value matrices plus output projection
- Bias terms are included in all calculations where applicable (+out_features term)