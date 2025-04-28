"""
Training script for the Variational Autoencoder (VAE).
"""
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle

from models.vae import build_encoder, build_decoder, VAE
from data.data_loader import train_generator, val_generator
from config.model_config import (
    LEARNING_RATE, VAE_EPOCHS, BATCH_SIZE,
    VAE_WEIGHTS_PATH, VAE_CHECKPOINTS_PATH
)

def train_vae():
    """
    Train the VAE model.

    Returns:
        tuple: (vae, encoder, decoder) - Trained models
    """
    # Build the encoder and decoder
    encoder = build_encoder()
    decoder = build_decoder()

    # Get the size of the training dataset
    train_data_size = len(train_generator.filenames)
    print(f"Training dataset size: {train_data_size} images")

    # Build the VAE model with data size for KL annealing
    vae = VAE(
        encoder=encoder,
        decoder=decoder,
        batch_size=BATCH_SIZE,
        train_data_size=train_data_size
    )
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))

    # Create directories for saving weights
    os.makedirs(VAE_WEIGHTS_PATH, exist_ok=True)
    os.makedirs(VAE_CHECKPOINTS_PATH, exist_ok=True)

    # Create checkpoint callback with correct filepath format
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(VAE_CHECKPOINTS_PATH, "vae_weights_epoch_{epoch:02d}.weights.h5"),
        save_weights_only=True,
        save_freq='epoch',
        verbose=1
    )

    # Custom callback to monitor validation metrics
    class ValidationMonitor(tf.keras.callbacks.Callback):
        def __init__(self, validation_data):
            super(ValidationMonitor, self).__init__()
            self.validation_data = validation_data

        def on_epoch_end(self, epoch, logs=None):
            # Get a batch of validation data
            val_batch = next(iter(self.validation_data))
            if isinstance(val_batch, tuple):
                val_batch = val_batch[0]  # Get just the images, not the labels

            # Forward pass
            z_mean, z_log_var, z = self.model.encoder(val_batch)
            reconstruction = self.model.decoder(z)

            # Calculate losses
            recon_loss = tf.reduce_mean(
                tf.keras.losses.mean_squared_error(val_batch, reconstruction)
            )
            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            )

            # Get current KL weight - convert to numpy for printing
            kl_weight = self.model.get_kl_weight()
            if isinstance(kl_weight, tf.Tensor):
                kl_weight = kl_weight.numpy()

            print(f"\nValidation metrics for epoch {epoch+1}:")
            print(f"  Reconstruction loss: {recon_loss:.6f}")
            print(f"  KL loss: {kl_loss:.6f}")
            print(f"  KL weight: {kl_weight:.6f}")
            print(f"  Total loss: {recon_loss + kl_weight * kl_loss:.6f}")

    validation_monitor = ValidationMonitor(val_generator)

    # Training VAE
    print(f"Training VAE for {VAE_EPOCHS} epochs...")
    history = vae.fit(
        train_generator,
        epochs=VAE_EPOCHS,
        validation_data=val_generator,
        callbacks=[checkpoint_callback, validation_monitor]
    )

    # Save the weights separately for encoder and decoder
    encoder_path = os.path.join(VAE_WEIGHTS_PATH, "encoder.weights.h5")
    decoder_path = os.path.join(VAE_WEIGHTS_PATH, "decoder.weights.h5")
    vae_path = os.path.join(VAE_WEIGHTS_PATH, "vae_complete.weights.h5")

    encoder.save_weights(encoder_path)
    decoder.save_weights(decoder_path)
    vae.save_weights(vae_path)

    # Save the complete model
    vae.save(os.path.join(VAE_WEIGHTS_PATH, "vae_complete_model.keras"))

    # Save the training history
    with open('vae_training_history.pkl', 'wb') as file:
        pickle.dump(history.history, file)

    print("VAE training completed. Models and weights saved.")

    return vae, encoder, decoder

if __name__ == "__main__":
    train_vae()
