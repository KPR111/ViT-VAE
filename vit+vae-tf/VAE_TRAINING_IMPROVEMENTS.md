# VAE Training Improvements

This document explains the improvements made to the VAE training process to address the issue of rapidly dropping KL loss.

## Problem Description

The original VAE implementation had the following issues:

1. **Rapid Drop in KL Loss**: The KL loss decreased too quickly during training, dropping from 1 to 0.0164 within the first epoch.
2. **Vanishing KL Loss**: By the second epoch, the KL loss dropped to zero (0.0000e+00).
3. **Zero Validation Losses**: All validation metrics (KL loss, reconstruction loss) showed zero values.
4. **Imbalanced Training Losses**: The model was over-regularizing the latent space at the cost of reconstruction quality.

## Implemented Solutions

### 1. Reduced KL Weight

The KL weight factor was reduced from 1.0 to 0.001 to prevent the model from overly focusing on the regularization term.

```python
# In config/model_config.py
KL_WEIGHT = 0.001  # Weight factor for KL loss in VAE (reduced from 1.0)
```

### 2. KL Weight Annealing

Implemented KL weight annealing to gradually increase the importance of the KL term during training:

```python
# In config/model_config.py
KL_ANNEALING_EPOCHS = 5  # Number of epochs to gradually increase KL weight
```

The VAE class now calculates the KL weight dynamically based on the current epoch:

```python
def get_kl_weight(self):
    """
    Calculate the current KL weight based on annealing schedule.
    Uses a linear annealing schedule from 0 to kl_weight_max over annealing_epochs.
    """
    # Get current epoch from optimizer iterations
    current_epoch = self.optimizer.iterations / (self.train_data_size // self.batch_size)
    
    # Linear annealing from 0 to kl_weight_max
    if current_epoch >= self.annealing_epochs:
        return self.kl_weight_max
    else:
        return (current_epoch / self.annealing_epochs) * self.kl_weight_max
```

### 3. Enhanced Model Architecture

The encoder and decoder architectures were improved with:

1. **Batch Normalization**: Added after each convolutional layer to stabilize training
2. **Increased Capacity**: Added more layers and increased the number of filters
3. **Proper Weight Initialization**: Used appropriate initializers for different layer types

### 4. Validation Monitoring

Added a custom callback to monitor validation metrics during training:

```python
class ValidationMonitor(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Get a batch of validation data
        val_batch = next(iter(self.validation_data))
        
        # Forward pass
        z_mean, z_log_var, z = self.model.encoder(val_batch)
        reconstruction = self.model.decoder(z)
        
        # Calculate losses
        recon_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(val_batch, reconstruction))
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        
        # Print detailed metrics
        print(f"\nValidation metrics for epoch {epoch+1}:")
        print(f"  Reconstruction loss: {recon_loss:.6f}")
        print(f"  KL loss: {kl_loss:.6f}")
        print(f"  KL weight: {self.model.get_kl_weight():.6f}")
        print(f"  Total loss: {recon_loss + self.model.get_kl_weight() * kl_loss:.6f}")
```

## How to Run the Improved Training

To train the VAE with these improvements:

```bash
# Navigate to the project directory
cd vit+vae

# Train the VAE for 10 epochs (or adjust as needed)
python main.py train-vae --epochs 10
```

## Expected Outcomes

With these improvements, you should observe:

1. **Balanced KL Loss**: The KL loss should decrease more gradually and stabilize at a non-zero value.
2. **Better Reconstruction Quality**: The model should achieve better reconstruction quality while maintaining a meaningful latent space.
3. **Proper Validation Metrics**: Validation metrics should show reasonable, non-zero values.
4. **Improved Latent Space**: The latent space should capture more meaningful features of the input data.

## Monitoring Training Progress

During training, the ValidationMonitor callback will print detailed metrics for each epoch, allowing you to track:

- Reconstruction loss
- KL loss
- Current KL weight (which increases gradually due to annealing)
- Total loss

This information will help you assess whether the training is progressing correctly.
