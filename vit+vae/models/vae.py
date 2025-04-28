"""
Variational Autoencoder (VAE) model implementation.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from config.model_config import IMAGE_SHAPE, LATENT_DIM, KL_WEIGHT

def build_encoder():
    """
    Build the VAE encoder model.

    Returns:
        Model: Encoder model that outputs [z_mean, z_log_var, z]
    """
    inputs = Input(shape=IMAGE_SHAPE, name="input_layer")

    x = layers.Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)

    z_mean = layers.Dense(LATENT_DIM, name='z_mean')(x)
    z_log_var = layers.Dense(LATENT_DIM, name='z_log_var')(x)

    # Sampling function (reparameterization trick)
    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(sampling, output_shape=(LATENT_DIM,), name='z')([z_mean, z_log_var])

    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    return encoder

def build_decoder():
    """
    Build the VAE decoder model.

    Returns:
        Model: Decoder model
    """
    latent_inputs = Input(shape=(LATENT_DIM,), name="latent_input")

    x = layers.Dense(56 * 56 * 32, activation='relu')(latent_inputs)
    x = layers.Reshape((56, 56, 32))(x)
    x = layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(16, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    outputs = layers.Conv2DTranspose(3, kernel_size=3, activation='sigmoid', padding='same')(x)

    decoder = Model(latent_inputs, outputs, name='decoder')
    return decoder

class VAE(Model):
    """
    Variational Autoencoder model that combines encoder and decoder.
    """
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
        # Apply KL weight factor
        weighted_kl_loss = KL_WEIGHT * kl_loss
        self.add_loss(weighted_kl_loss)
        return reconstructed

    def compile(self, optimizer):
        super(VAE, self).compile()
        self.optimizer = optimizer
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.weighted_kl_loss_tracker = tf.keras.metrics.Mean(name="weighted_kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.weighted_kl_loss_tracker,
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
            # Apply KL weight factor
            weighted_kl_loss = KL_WEIGHT * kl_loss
            total_loss = reconstruction_loss + weighted_kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.weighted_kl_loss_tracker.update_state(weighted_kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "weighted_kl_loss": self.weighted_kl_loss_tracker.result(),
        }

# Only build and test models when this file is run directly
if __name__ == "__main__":
    # Build
    encoder = build_encoder()
    decoder = build_decoder()

    # Test models
    print("VAE Encoder Summary:")
    encoder.summary()
    print("\nVAE Decoder Summary:")
    decoder.summary()
