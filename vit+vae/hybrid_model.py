from tensorflow.keras import Model, Input
import tensorflow as tf
import os
from load_vae import load_vae_encoder, verify_vae_weights
from models.vit_models import build_vit_classifier
from loadingpreprocessing import train_generator, val_generator

# First verify VAE weights exist
if not verify_vae_weights():
    print("Please train VAE first using trainvae.py")
    exit()

# Create directories for saving weights
os.makedirs("saved_models/hybrid", exist_ok=True)
os.makedirs("checkpoints/hybrid", exist_ok=True)

# Load pre-trained encoder
encoder = load_vae_encoder()
if encoder is None:
    print("Failed to load VAE encoder. Please train VAE first.")
    exit()

# Build ViT classifier
vit_classifier = build_vit_classifier()

# Freeze VAE Encoder weights
encoder.trainable = False
print("VAE Encoder layers frozen")

# Full Hybrid Model
def build_hybrid_model(encoder, vit_classifier):
    inputs = Input(shape=(224, 224, 3))

    # Get latent vector from encoder (we only need the z vector)
    z_mean, z_log_var, z = encoder(inputs)

    # Pass through Vision Transformer
    outputs = vit_classifier(z)

    model = Model(inputs, outputs, name="vae_vit_hybrid")
    return model

# Build and compile
hybrid_model = build_hybrid_model(encoder, vit_classifier)
hybrid_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Create checkpoint callback
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="checkpoints/hybrid/hybrid_model_epoch_{epoch:02d}.weights.h5",  # Added .weights before .h5
    save_weights_only=True,
    save_freq='epoch',
    verbose=1
)

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Train
history = hybrid_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=1,#update epochs
    callbacks=[checkpoint_callback, early_stopping]
)

# Save the final hybrid model weights
hybrid_model.save_weights("saved_models/hybrid/hybrid_final.weights.h5")  # Added .weights before .h5

# Save the complete model
hybrid_model.save("saved_models/hybrid_complete_model.keras")

# Save the training history
import pickle
with open('training_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)

print("Training completed. Model and history saved.")
