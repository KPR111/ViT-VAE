"""
Training script for the hybrid VAE-ViT model.
"""
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pickle

from models.vae import build_encoder
from models.vit import build_vit_classifier
from models.hybrid import build_hybrid_model
from data.data_loader import train_generator, val_generator
from utils.model_utils import load_vae_encoder, verify_vae_weights
from config.model_config import (
    LEARNING_RATE, HYBRID_EPOCHS,
    HYBRID_WEIGHTS_PATH, HYBRID_CHECKPOINTS_PATH
)

def train_hybrid_model():
    """
    Train the hybrid VAE-ViT model.
    
    Returns:
        Model: Trained hybrid model or None if training fails
    """
    # First verify VAE weights exist
    if not verify_vae_weights():
        print("Please train VAE first using train_vae.py")
        return None
    
    # Create directories for saving weights
    os.makedirs(HYBRID_WEIGHTS_PATH, exist_ok=True)
    os.makedirs(HYBRID_CHECKPOINTS_PATH, exist_ok=True)
    
    # Load pre-trained encoder
    encoder = load_vae_encoder()
    if encoder is None:
        print("Failed to load VAE encoder. Please train VAE first.")
        return None
    
    # Build ViT classifier
    vit_classifier = build_vit_classifier()
    
    # Freeze VAE Encoder weights
    encoder.trainable = False
    print("VAE Encoder layers frozen")
    
    # Build and compile
    hybrid_model = build_hybrid_model(encoder, vit_classifier)
    hybrid_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(HYBRID_CHECKPOINTS_PATH, "hybrid_model_epoch_{epoch:02d}.weights.h5"),
        save_weights_only=True,
        save_freq='epoch',
        verbose=1
    )
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    # Train
    print(f"Training hybrid model for {HYBRID_EPOCHS} epochs...")
    history = hybrid_model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=HYBRID_EPOCHS,
        callbacks=[checkpoint_callback, early_stopping]
    )
    
    # Save the final hybrid model weights
    weights_path = os.path.join(HYBRID_WEIGHTS_PATH, "hybrid_final.weights.h5")
    hybrid_model.save_weights(weights_path)
    
    # Save the complete model
    model_path = os.path.join(HYBRID_WEIGHTS_PATH, "hybrid_complete_model.keras")
    hybrid_model.save(model_path)
    
    # Save the training history
    with open('hybrid_training_history.pkl', 'wb') as file:
        pickle.dump(history.history, file)
    
    print("Hybrid model training completed. Model and history saved.")
    
    return hybrid_model

if __name__ == "__main__":
    train_hybrid_model()
