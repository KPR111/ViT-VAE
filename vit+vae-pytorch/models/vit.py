"""
Vision Transformer (ViT) model implementation in PyTorch.
"""
import torch
import torch.nn as nn
from config.model_config import (
    LATENT_DIM, NUM_CLASSES, PATCH_SIZE, NUM_PATCHES,
    NUM_TRANSFORMER_LAYERS, NUM_HEADS, DROPOUT_RATE
)

class MultiLayerPerceptron(nn.Module):
    """
    Multilayer Perceptron block for Vision Transformer.
    """
    def __init__(self, hidden_units, dropout_rate=0.1):
        super(MultiLayerPerceptron, self).__init__()
        self.layers = nn.ModuleList()
        
        for units in hidden_units:
            self.layers.append(nn.Linear(PATCH_SIZE, units))
            self.layers.append(nn.GELU())
            self.layers.append(nn.Dropout(dropout_rate))
            
            # Update input size for next layer
            PATCH_SIZE = units
    
    def forward(self, x):
        """
        Forward pass through the MLP.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor: Output tensor
        """
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerBlock(nn.Module):
    """
    Transformer block for Vision Transformer.
    """
    def __init__(self, embed_dim, num_heads, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # MLP block
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x):
        """
        Forward pass through the transformer block.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
            
        Returns:
            Tensor: Output tensor of shape [batch_size, seq_len, embed_dim]
        """
        # Layer normalization and self-attention
        x_norm = self.layer_norm1(x)
        attention_output, _ = self.attention(x_norm, x_norm, x_norm)
        
        # Skip connection
        x = x + attention_output
        
        # Layer normalization and MLP
        x_norm = self.layer_norm2(x)
        mlp_output = self.mlp(x_norm)
        
        # Skip connection
        x = x + mlp_output
        
        return x

class VisionTransformer(nn.Module):
    """
    Vision Transformer classifier model.
    """
    def __init__(self):
        super(VisionTransformer, self).__init__()
        
        # Initial dense layer to expand the latent vector
        self.dense = nn.Linear(LATENT_DIM, LATENT_DIM)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=PATCH_SIZE,
                num_heads=NUM_HEADS,
                dropout_rate=DROPOUT_RATE
            ) for _ in range(NUM_TRANSFORMER_LAYERS)
        ])
        
        # Final layer normalization
        self.layer_norm = nn.LayerNorm(PATCH_SIZE, eps=1e-6)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(DROPOUT_RATE * 3),  # Higher dropout at the end
            nn.Linear(NUM_PATCHES * PATCH_SIZE, NUM_CLASSES),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        """
        Forward pass through the Vision Transformer.
        
        Args:
            x: Input tensor of shape [batch_size, latent_dim]
            
        Returns:
            Tensor: Class probabilities of shape [batch_size, num_classes]
        """
        # Expand the latent vector
        x = self.dense(x)
        
        # Reshape to patches
        x = x.view(x.size(0), NUM_PATCHES, PATCH_SIZE)
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # Classification
        x = self.classifier(x)
        
        return x

def build_vit_classifier():
    """
    Build the Vision Transformer classifier model.
    
    Returns:
        VisionTransformer: ViT classifier model
    """
    return VisionTransformer()

# Only build and test model when this file is run directly
if __name__ == "__main__":
    # Build
    vit_classifier = build_vit_classifier()
    
    # Test with random input
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit_classifier = vit_classifier.to(device)
    
    # Create random input (latent vector)
    batch_size = 4
    x = torch.randn(batch_size, LATENT_DIM).to(device)
    
    # Forward pass
    output = vit_classifier(x)
    
    # Print shapes
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output sum (should be close to 1.0 for each sample): {output.sum(dim=1)}")
