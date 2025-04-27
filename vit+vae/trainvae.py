# Import
import tensorflow as tf
from tensorflow.keras import Model
import os
from tensorflow.keras.callbacks import ModelCheckpoint

from models.vae_models import build_encoder, build_decoder
from loadingpreprocessing import train_generator, val_generator

encoder = build_encoder()
decoder = build_decoder()


# VAE Model (Wrapper)
class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.mse_loss_fn = tf.keras.losses.MeanSquaredError()

    def call(self, inputs, training=None):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL loss as part of the layer's loss
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        self.add_loss(kl_loss)
        return reconstructed


    def compile(self, optimizer):
        super(VAE, self).compile()
        self.optimizer = optimizer
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = self.mse_loss_fn(data, reconstruction)


            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

# Build the VAE model
vae = VAE(encoder, decoder)
vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

# Create directories for saving weights if they don't exist
checkpoint_dir = "checkpoints/vae"
final_model_dir = "saved_models"
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(final_model_dir, exist_ok=True)

# Create checkpoint callback
checkpoint_path = os.path.join(checkpoint_dir, "vae_weights_epoch_{epoch:02d}.h5")
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    save_freq='epoch',  # save every epoch
    verbose=1
)

# Training VAE
vae.fit(
    train_generator,
    epochs=15,  # You can increase later
    validation_data=val_generator,
    callbacks=[checkpoint_callback]
)

# Save the final model weights
final_weights_path = os.path.join(final_model_dir, "vae_final_weights.h5")
vae.save_weights(final_weights_path)
print(f"Final weights saved to {final_weights_path}")

# Optional: Save the entire model (architecture + weights)
final_model_path = os.path.join(final_model_dir, "vae_complete_model")
vae.save(final_model_path)
print(f"Complete model saved to {final_model_path}")



