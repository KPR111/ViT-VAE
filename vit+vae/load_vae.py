# Load weights into a new model instance
new_vae = VAE(encoder, decoder)
new_vae.load_weights("saved_models/vae_final_weights.h5")

# Or load the complete model
loaded_vae = tf.keras.models.load_model("saved_models/vae_complete_model")