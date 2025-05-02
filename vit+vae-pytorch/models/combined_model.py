"""
Combined VAE-ViT model implementation in PyTorch.
This model trains both the VAE and ViT components simultaneously.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.model_config import (
    IMAGE_SIZE, IMAGE_CHANNELS, LATENT_DIM,
    KL_WEIGHT, KL_ANNEALING_EPOCHS
)

class CombinedModel(nn.Module):
    """
    Combined model that includes both VAE and ViT components.
    Both components are trained simultaneously.
    """
    def __init__(self, encoder, decoder, vit_classifier, batch_size, train_data_size):
        super(CombinedModel, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.vit_classifier = vit_classifier

        # VAE parameters for KL annealing
        self.batch_size = batch_size
        self.train_data_size = train_data_size #70,304
        self.steps_per_epoch = self.train_data_size // self.batch_size #70304/32=2197
        self.current_epoch = 0
        self.current_iteration = 0
        self.annealing_epochs = KL_ANNEALING_EPOCHS  # Number of epochs to gradually increase KL weight
        self.kl_weight_max = KL_WEIGHT  # Maximum KL weight

    def forward(self, x):
        """
        Forward pass through the combined model.

        Args:
            x: Input tensor of shape [batch_size, channels, height, width]

        Returns:
            tuple: (reconstructed, class_outputs, mean, log_var, z)
        """
        # VAE encoding
        mean, log_var, z = self.encoder(x)

        # VAE decoding
        reconstructed = self.decoder(z)

        # ViT classification
        class_outputs = self.vit_classifier(z)

        return reconstructed, class_outputs, mean, log_var, z

    def get_kl_weight(self):
        """
        Get the current KL weight with annealing.

        Returns:
            float: Current KL weight
        """
        # Calculate current epoch as a float
        current_epoch = self.current_iteration / self.steps_per_epoch

        # Apply annealing
        if current_epoch >= self.annealing_epochs:
            return self.kl_weight_max
        else:
            return (current_epoch / self.annealing_epochs) * self.kl_weight_max

    def loss_function(self, recon_x, x, class_outputs, targets, mean, log_var):
        """
        Calculate the combined loss function.

        Args:
            recon_x: Reconstructed input
            x: Original input
            class_outputs: Class predictions from ViT
            targets: Ground truth class labels
            mean: Mean of the latent distribution
            log_var: Log variance of the latent distribution

        Returns:
            tuple: (total_loss, reconstruction_loss, classification_loss, kl_loss, weighted_kl_loss, kl_weight)
        """
        # Reconstruction loss (mean squared error)
        reconstruction_loss = F.mse_loss(recon_x, x, reduction='mean')

        # Classification loss (cross entropy)
        classification_loss = F.cross_entropy(class_outputs, targets)

        # KL divergence
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())

        # Get current KL weight with annealing
        kl_weight = self.get_kl_weight()
        weighted_kl_loss = kl_weight * kl_loss

        # Total loss: reconstruction + classification + weighted KL
        total_loss = reconstruction_loss + classification_loss + weighted_kl_loss

        return total_loss, reconstruction_loss, classification_loss, kl_loss, weighted_kl_loss, kl_weight

def build_combined_model(encoder, decoder, vit_classifier, batch_size, train_data_size):
    """
    Build the combined model that includes both VAE and ViT components.

    Args:
        encoder: VAE encoder model
        decoder: VAE decoder model
        vit_classifier: Vision Transformer classifier model
        batch_size: Batch size for KL annealing
        train_data_size: Size of training dataset for KL annealing

    Returns:
        CombinedModel: Combined model
    """
    return CombinedModel(encoder, decoder, vit_classifier, batch_size, train_data_size)

# Only build and test model when this file is run directly
if __name__ == "__main__":
    from vae import build_encoder, build_decoder
    from vit import build_vit_classifier

    # Build the base models
    encoder = build_encoder()
    decoder = build_decoder()
    vit_classifier = build_vit_classifier()

    # Build the combined model
    from config.model_config import BATCH_SIZE
    batch_size = BATCH_SIZE
    train_data_size = 1000  # Example value for testing
    combined_model = build_combined_model(encoder, decoder, vit_classifier, batch_size, train_data_size)

    # Test with random input
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    combined_model = combined_model.to(device)

    # Create random input
    test_batch_size = 4
    from config.model_config import NUM_CLASSES
    x = torch.randn(test_batch_size, IMAGE_CHANNELS, IMAGE_SIZE[0], IMAGE_SIZE[1]).to(device)
    targets = torch.randint(0, NUM_CLASSES, (test_batch_size,)).to(device)

    # Forward pass
    reconstructed, class_outputs, mean, log_var, z = combined_model(x)

    # Calculate loss
    total_loss, recon_loss, class_loss, kl_loss, weighted_kl, kl_weight = combined_model.loss_function(
        reconstructed, x, class_outputs, targets, mean, log_var
    )

    # Print shapes and losses
    print(f"Input shape: {x.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Class outputs shape: {class_outputs.shape}")
    print(f"Latent vector shape: {z.shape}")
    print(f"Total loss: {total_loss.item()}")
    print(f"Reconstruction loss: {recon_loss.item()}")
    print(f"Classification loss: {class_loss.item()}")
    print(f"KL loss: {kl_loss.item()}")
    print(f"Weighted KL loss: {weighted_kl.item()}")
    print(f"KL weight: {kl_weight}")
