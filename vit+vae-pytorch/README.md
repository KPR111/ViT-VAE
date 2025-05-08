# Hybrid VAE-ViT Model for Plant Disease Classification

## Model Architecture Parameters

### VAE Model Parameters
Total parameters: 32,456,195
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
Total parameters: 84,134
- Initial Dense Layer: 65,792 parameters
  - Calculation: (LATENT_DIM × LATENT_DIM) + LATENT_DIM
  - (256 × 256) + 256 = 65,792

- Classification Head: 18,342 parameters
  - Calculation: (LATENT_DIM × NUM_CLASSES) + NUM_CLASSES
  - (256 × 38) + 38 = 18,342

### Combined Model
Total trainable parameters: 32,540,329 parameters

### Notes on Parameter Calculations:
- For convolutional layers: parameters = (kernel_size² × in_channels × out_channels) + out_channels
- For fully connected layers: parameters = (input_features × output_features) + output_features
- For layer normalization: parameters = 2 × features (gamma and beta parameters)
- For multi-head attention: parameters include query, key, and value matrices plus output projection
- Bias terms are included in all calculations where applicable (+out_features term)