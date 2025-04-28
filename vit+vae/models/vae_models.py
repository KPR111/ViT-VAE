# Import
import tensorflow as tf
from tensorflow.keras import layers, Model, Input

# (then your build_encoder and build_decoder functions...)

# VAE Constants
IMAGE_SHAPE = (224, 224, 3)
LATENT_DIM = 256

# Encoder
def build_encoder():
    inputs = Input(shape=IMAGE_SHAPE)

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

# Decoder
def build_decoder():
    latent_inputs = Input(shape=(LATENT_DIM,))

    x = layers.Dense(56 * 56 * 32, activation='relu')(latent_inputs)
    x = layers.Reshape((56, 56, 32))(x)
    x = layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(16, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    outputs = layers.Conv2DTranspose(3, kernel_size=3, activation='sigmoid', padding='same')(x)

    decoder = Model(latent_inputs, outputs, name='decoder')
    return decoder

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
