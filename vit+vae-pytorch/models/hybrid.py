"""
Hybrid model that combines VAE and ViT for plant disease classification in PyTorch.
"""
import torch
import torch.nn as nn
from config.model_config import IMAGE_SIZE, IMAGE_CHANNELS

class HybridModel(nn.Module):
    """
    Hybrid model that combines VAE encoder and ViT classifier.
    """
    def __init__(self, encoder, vit_classifier):
        super(HybridModel, self).__init__()
        
        self.encoder = encoder
        self.vit_classifier = vit_classifier
    
    def forward(self, x):
        """
        Forward pass through the hybrid model.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Tensor: Class probabilities of shape [batch_size, num_classes]
        """
        # Get latent vector from encoder (we only need the z vector)
        mean, log_var, z = self.encoder(x)
        
        # Pass through Vision Transformer
        outputs = self.vit_classifier(z)
        
        return outputs

def build_hybrid_model(encoder, vit_classifier):
    """
    Build the hybrid model that combines VAE encoder and ViT classifier.
    
    Args:
        encoder: VAE encoder model
        vit_classifier: Vision Transformer classifier model
        
    Returns:
        HybridModel: Hybrid model
    """
    return HybridModel(encoder, vit_classifier)

# Only build and test model when this file is run directly
if __name__ == "__main__":
    from vae import build_encoder
    from vit import build_vit_classifier
    
    # Build the base models
    encoder = build_encoder()
    vit_classifier = build_vit_classifier()
    
    # Build the hybrid model
    hybrid_model = build_hybrid_model(encoder, vit_classifier)
    
    # Test with random input
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hybrid_model = hybrid_model.to(device)
    
    # Create random input
    batch_size = 4
    x = torch.randn(batch_size, IMAGE_CHANNELS, IMAGE_SIZE[0], IMAGE_SIZE[1]).to(device)
    
    # Forward pass
    output = hybrid_model(x)
    
    # Print shapes
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output sum (should be close to 1.0 for each sample): {output.sum(dim=1)}")
