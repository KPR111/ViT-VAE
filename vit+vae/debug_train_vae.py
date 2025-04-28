"""
Debug version of the VAE training script with additional logging and error handling.
"""
import os
import tensorflow as tf
import time
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle
import sys

# Set TensorFlow logging level
tf.get_logger().setLevel('INFO')

def train_vae_debug(epochs=5, batch_size=16):
    """
    Debug version of the VAE training function with more verbose output.
    
    Args:
        epochs (int): Number of epochs to train
        batch_size (int): Batch size for training
        
    Returns:
        tuple: (vae, encoder, decoder) - Trained models
    """
    print("\n=== Debug VAE Training ===")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    
    # Import here to avoid circular imports
    from models.vae import build_encoder, build_decoder, VAE
    from config.model_config import TRAIN_DIR, VALID_DIR, IMAGE_SHAPE, LEARNING_RATE
    
    # Check if data directories exist
    print("\nChecking data directories...")
    if not os.path.exists(TRAIN_DIR):
        print(f"ERROR: Training directory not found: {TRAIN_DIR}")
        return None, None, None
    if not os.path.exists(VALID_DIR):
        print(f"ERROR: Validation directory not found: {VALID_DIR}")
        return None, None, None
    
    print("Data directories found.")
    
    # Create data generators
    print("\nCreating data generators...")
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    # Data preprocessing (augmentation only for train)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=10,
        zoom_range=0.1
    )

    valid_datagen = ImageDataGenerator(
        rescale=1./255
    )
    
    # Create train generator
    print("Creating training generator...")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SHAPE[:2],
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    # Create validation generator
    print("Creating validation generator...")
    val_generator = valid_datagen.flow_from_directory(
        VALID_DIR,
        target_size=IMAGE_SHAPE[:2],
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Build the encoder and decoder
    print("\nBuilding models...")
    encoder = build_encoder()
    decoder = build_decoder()
    
    # Build the VAE model
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))
    
    # Create directories for saving weights
    print("\nSetting up directories for checkpoints...")
    VAE_WEIGHTS_PATH = "saved_models/vae"
    VAE_CHECKPOINTS_PATH = "checkpoints/vae"
    
    os.makedirs(VAE_WEIGHTS_PATH, exist_ok=True)
    os.makedirs(VAE_CHECKPOINTS_PATH, exist_ok=True)
    
    # Create checkpoint callback with correct filepath format
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(VAE_CHECKPOINTS_PATH, "vae_weights_epoch_{epoch:02d}.weights.h5"),
        save_weights_only=True,
        save_freq='epoch',
        verbose=1
    )
    
    # Custom callback to print progress
    class DebugCallback(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            print(f"\nStarting epoch {epoch+1}/{epochs}...")
            self.epoch_start_time = time.time()
            
        def on_epoch_end(self, epoch, logs=None):
            time_taken = time.time() - self.epoch_start_time
            print(f"Epoch {epoch+1}/{epochs} completed in {time_taken:.2f} seconds")
            print(f"Metrics: {logs}")
            
        def on_batch_end(self, batch, logs=None):
            if batch % 100 == 0:
                print(f"  Batch {batch} completed. Loss: {logs.get('loss', 'N/A')}")
    
    debug_callback = DebugCallback()
    
    # Training VAE
    print(f"\nStarting VAE training for {epochs} epochs...")
    try:
        history = vae.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=[checkpoint_callback, debug_callback],
            verbose=1
        )
        
        print("\nTraining completed successfully!")
        
        # Save the weights separately for encoder and decoder
        print("\nSaving model weights...")
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
        
    except Exception as e:
        print(f"\nERROR during training: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    # Get command line arguments
    epochs = 5  # default
    batch_size = 16  # default
    
    if len(sys.argv) > 1:
        try:
            epochs = int(sys.argv[1])
        except:
            pass
    
    if len(sys.argv) > 2:
        try:
            batch_size = int(sys.argv[2])
        except:
            pass
    
    print(f"Starting debug VAE training with {epochs} epochs and batch size {batch_size}")
    train_vae_debug(epochs=epochs, batch_size=batch_size)
