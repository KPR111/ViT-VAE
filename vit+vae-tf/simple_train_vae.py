"""
Simplified training script for the Variational Autoencoder (VAE).
This version removes the validation monitor to avoid the error.
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

def train_vae_simple():
    """
    Train the VAE model with simplified monitoring.
    
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

    # Training VAE
    print(f"Training VAE for {VAE_EPOCHS} epochs...")
    history = vae.fit(
        train_generator,
        epochs=VAE_EPOCHS,
        validation_data=val_generator,
        callbacks=[checkpoint_callback]
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
    train_vae_simple()
