"""
Variational Autoencoder (VAE) model implementation in PyTorch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.model_config import IMAGE_SIZE, IMAGE_CHANNELS, LATENT_DIM, KL_WEIGHT, KL_ANNEALING_EPOCHS

class Encoder(nn.Module):
    """
    VAE encoder network.
    """
    def __init__(self):
        super(Encoder, self).__init__()

        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(IMAGE_CHANNELS, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Calculate the size of the flattened features
        # After 3 layers with stride 2, the size is reduced by factor of 2^3 = 8
        self.feature_size = (IMAGE_SIZE[0] // 8) * (IMAGE_SIZE[1] // 8) * 128

        # Fully connected layers
        self.fc = nn.Linear(self.feature_size, 256)
        self.bn_fc = nn.BatchNorm1d(256)

        # Latent space
        self.fc_mean = nn.Linear(256, LATENT_DIM)
        self.fc_logvar = nn.Linear(256, LATENT_DIM)

        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize the weights using Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through the encoder.

        Args:
            x: Input tensor of shape [batch_size, channels, height, width]

        Returns:
            tuple: (mean, log_var, z) where z is the sampled latent vector
        """
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layer
        x = F.relu(self.bn_fc(self.fc(x)))

        # Get mean and log variance
        mean = self.fc_mean(x)
        log_var = self.fc_logvar(x)

        # Clip log_var to prevent numerical instability
        log_var = torch.clamp(log_var, -20.0, 2.0)

        # Reparameterization trick
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mean + eps * std

        return mean, log_var, z

class Decoder(nn.Module):
    """
    VAE decoder network.
    """
    def __init__(self):
        super(Decoder, self).__init__()

        # Calculate the initial size for the decoder
        # After 3 layers with stride 2, the size is reduced by factor of 2^3 = 8
        self.initial_size = (IMAGE_SIZE[0] // 8, IMAGE_SIZE[1] // 8)
        self.initial_channels = 32

        # Fully connected layer to convert latent vector to feature map
        self.fc = nn.Linear(LATENT_DIM, self.initial_size[0] * self.initial_size[1] * self.initial_channels)
        self.bn_fc = nn.BatchNorm1d(self.initial_size[0] * self.initial_size[1] * self.initial_channels)

        # Transposed convolutional layers with batch normalization
        self.deconv1 = nn.ConvTranspose2d(self.initial_channels, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(16)

        # Final layer to produce image
        self.deconv_final = nn.ConvTranspose2d(16, IMAGE_CHANNELS, kernel_size=3, padding=1)

        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize the weights using He initialization for ReLU activations."""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        """
        Forward pass through the decoder.

        Args:
            z: Latent vector of shape [batch_size, latent_dim]

        Returns:
            Tensor: Reconstructed image
        """
        # Fully connected layer
        x = F.relu(self.bn_fc(self.fc(z)))

        # Reshape to feature map -> Just Reorganizing
        x = x.view(x.size(0), self.initial_channels, self.initial_size[0], self.initial_size[1])

        # Transposed convolutional layers
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))

        # Final layer with sigmoid activation for pixel values between 0 and 1
        x = torch.sigmoid(self.deconv_final(x))

        return x

class VAE(nn.Module):
    """
    Variational Autoencoder model that combines encoder and decoder.
    """
    def __init__(self, batch_size=32, train_data_size=None):
        super(VAE, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.batch_size = batch_size
        self.train_data_size = train_data_size or 10000  # Default value if not provided
        self.kl_weight_max = KL_WEIGHT  # Store the maximum KL weight
        self.annealing_epochs = KL_ANNEALING_EPOCHS  # Number of epochs for annealing

        # For tracking metrics
        self.current_epoch = 0
        self.steps_per_epoch = self.train_data_size // self.batch_size
        self.current_iteration = 0

    def forward(self, x):
        """
        Forward pass through the VAE.

        Args:
            x: Input tensor of shape [batch_size, channels, height, width]

        Returns:
            tuple: (reconstructed, mean, log_var, z)
        """
        mean, log_var, z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed, mean, log_var, z

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

    def loss_function(self, recon_x, x, mean, log_var):
        """
        Calculate the VAE loss function.

        Args:
            recon_x: Reconstructed input
            x: Original input
            mean: Mean of the latent distribution
            log_var: Log variance of the latent distribution

        Returns:
            tuple: (total_loss, reconstruction_loss, kl_loss, weighted_kl_loss, kl_weight)
        """
        # Reconstruction loss (mean squared error)
        reconstruction_loss = F.mse_loss(recon_x, x, reduction='mean')

        # KL divergence
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())

        # Get current KL weight with annealing
        kl_weight = self.get_kl_weight()
        weighted_kl_loss = kl_weight * kl_loss

        # Total loss
        total_loss = reconstruction_loss + weighted_kl_loss

        return total_loss, reconstruction_loss, kl_loss, weighted_kl_loss, kl_weight

def build_encoder():
    """
    Build the VAE encoder model.

    Returns:
        Encoder: Encoder model
    """
    return Encoder()

def build_decoder():
    """
    Build the VAE decoder model.

    Returns:
        Decoder: Decoder model
    """
    return Decoder()

# Only build and test models when this file is run directly
if __name__ == "__main__":
    # Build
    encoder = build_encoder()
    decoder = build_decoder()
    vae = VAE()

    # Test with random input
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = vae.to(device)

    # Create random input
    batch_size = 4
    x = torch.randn(batch_size, IMAGE_CHANNELS, IMAGE_SIZE[0], IMAGE_SIZE[1]).to(device)

    # Forward pass
    reconstructed, mean, log_var, z = vae(x)

    # Calculate loss
    loss, recon_loss, kl_loss, weighted_kl, kl_weight = vae.loss_function(reconstructed, x, mean, log_var)

    # Print shapes and loss
    print(f"Input shape: {x.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Latent vector shape: {z.shape}")
    print(f"Loss: {loss.item():.4f} (Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f}, Weighted KL: {weighted_kl.item():.4f})")
    print(f"KL Weight: {kl_weight:.6f}")
