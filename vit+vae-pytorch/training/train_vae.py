"""
Training script for the Variational Autoencoder (VAE) in PyTorch.
"""
import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import pickle
from tqdm import tqdm

from models.vae import build_encoder, build_decoder, VAE
from data.data_loader import initialize_data_loaders
from utils.model_utils import save_checkpoint, load_checkpoint, get_device
from config.model_config import (
    LEARNING_RATE, VAE_EPOCHS, BATCH_SIZE,
    VAE_WEIGHTS_PATH, VAE_CHECKPOINTS_PATH
)

def train_vae(epochs=VAE_EPOCHS, resume=False):
    """
    Train the VAE model.
    
    Args:
        epochs (int): Number of epochs to train
        resume (bool): Whether to resume training from a checkpoint
        
    Returns:
        tuple: (vae, encoder, decoder) - Trained models
    """
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Initialize data loaders
    print("Initializing data loaders...")
    train_loader, val_loader, _ = initialize_data_loaders(verbose=True)
    
    # Build the encoder and decoder
    print("Building models...")
    encoder = build_encoder()
    decoder = build_decoder()
    
    # Get the size of the training dataset
    train_data_size = len(train_loader.dataset)
    print(f"Training dataset size: {train_data_size} images")
    
    # Build the VAE model with data size for KL annealing
    vae = VAE(
        batch_size=BATCH_SIZE,
        train_data_size=train_data_size
    )
    
    # Move models to device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    vae = vae.to(device)
    
    # Create optimizer
    optimizer = optim.Adam(vae.parameters(), lr=LEARNING_RATE)
    
    # Create directories for saving weights
    os.makedirs(VAE_WEIGHTS_PATH, exist_ok=True)
    os.makedirs(VAE_CHECKPOINTS_PATH, exist_ok=True)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(VAE_CHECKPOINTS_PATH, "logs"))
    
    # Starting epoch and best loss
    start_epoch = 0
    best_loss = float('inf')
    
    # Resume training if requested
    if resume:
        checkpoint_path = os.path.join(VAE_CHECKPOINTS_PATH, "vae_latest.pt")
        vae, optimizer, start_epoch, best_loss = load_checkpoint(vae, optimizer, checkpoint_path)
    
    # Training loop
    print(f"Training VAE for {epochs} epochs...")
    for epoch in range(start_epoch, epochs):
        # Training phase
        vae.train()
        train_loss = 0
        train_recon_loss = 0
        train_kl_loss = 0
        train_weighted_kl_loss = 0
        
        # Update current epoch in VAE for KL annealing
        vae.current_epoch = epoch
        
        # Progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (data, _) in enumerate(progress_bar):
            # Move data to device
            data = data.to(device)
            
            # Update current iteration in VAE for KL annealing
            vae.current_iteration = epoch * len(train_loader) + batch_idx
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            recon_batch, mean, log_var, z = vae(data)
            
            # Calculate loss
            loss, recon_loss, kl_loss, weighted_kl, kl_weight = vae.loss_function(recon_batch, data, mean, log_var)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()
            train_weighted_kl_loss += weighted_kl.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'recon': f"{recon_loss.item():.4f}",
                'kl': f"{kl_loss.item():.4f}",
                'w_kl': f"{weighted_kl.item():.4f}",
                'kl_w': f"{kl_weight:.6f}"
            })
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_train_recon_loss = train_recon_loss / len(train_loader)
        avg_train_kl_loss = train_kl_loss / len(train_loader)
        avg_train_weighted_kl_loss = train_weighted_kl_loss / len(train_loader)
        
        # Validation phase
        vae.eval()
        val_loss = 0
        val_recon_loss = 0
        val_kl_loss = 0
        val_weighted_kl_loss = 0
        
        with torch.no_grad():
            for data, _ in val_loader:
                # Move data to device
                data = data.to(device)
                
                # Forward pass
                recon_batch, mean, log_var, z = vae(data)
                
                # Calculate loss
                loss, recon_loss, kl_loss, weighted_kl, _ = vae.loss_function(recon_batch, data, mean, log_var)
                
                # Update metrics
                val_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_kl_loss += kl_loss.item()
                val_weighted_kl_loss += weighted_kl.item()
        
        # Calculate average losses
        avg_val_loss = val_loss / len(val_loader)
        avg_val_recon_loss = val_recon_loss / len(val_loader)
        avg_val_kl_loss = val_kl_loss / len(val_loader)
        avg_val_weighted_kl_loss = val_weighted_kl_loss / len(val_loader)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.4f} (Recon: {avg_train_recon_loss:.4f}, KL: {avg_train_kl_loss:.4f}, W_KL: {avg_train_weighted_kl_loss:.4f}), "
              f"Val Loss: {avg_val_loss:.4f} (Recon: {avg_val_recon_loss:.4f}, KL: {avg_val_kl_loss:.4f}, W_KL: {avg_val_weighted_kl_loss:.4f})")
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('ReconLoss/train', avg_train_recon_loss, epoch)
        writer.add_scalar('ReconLoss/val', avg_val_recon_loss, epoch)
        writer.add_scalar('KLLoss/train', avg_train_kl_loss, epoch)
        writer.add_scalar('KLLoss/val', avg_val_kl_loss, epoch)
        writer.add_scalar('WeightedKLLoss/train', avg_train_weighted_kl_loss, epoch)
        writer.add_scalar('WeightedKLLoss/val', avg_val_weighted_kl_loss, epoch)
        writer.add_scalar('KLWeight', kl_weight, epoch)
        
        # Save checkpoint for every epoch
        checkpoint_path = os.path.join(VAE_CHECKPOINTS_PATH, f"vae_epoch_{epoch+1:02d}.pt")
        save_checkpoint(vae, optimizer, epoch+1, avg_val_loss, checkpoint_path)
        
        # Save latest checkpoint (for resuming)
        latest_checkpoint_path = os.path.join(VAE_CHECKPOINTS_PATH, "vae_latest.pt")
        save_checkpoint(vae, optimizer, epoch+1, avg_val_loss, latest_checkpoint_path)
        
        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_checkpoint_path = os.path.join(VAE_CHECKPOINTS_PATH, "vae_best.pt")
            save_checkpoint(vae, optimizer, epoch+1, avg_val_loss, best_checkpoint_path)
            print(f"New best model saved with validation loss: {best_loss:.4f}")
    
    # Save the weights separately for encoder and decoder
    encoder_path = os.path.join(VAE_WEIGHTS_PATH, "encoder.pt")
    decoder_path = os.path.join(VAE_WEIGHTS_PATH, "decoder.pt")
    vae_path = os.path.join(VAE_WEIGHTS_PATH, "vae_complete.pt")
    
    torch.save(encoder.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)
    torch.save(vae.state_dict(), vae_path)
    
    # Save the training history
    history = {
        'train_loss': avg_train_loss,
        'train_recon_loss': avg_train_recon_loss,
        'train_kl_loss': avg_train_kl_loss,
        'train_weighted_kl_loss': avg_train_weighted_kl_loss,
        'val_loss': avg_val_loss,
        'val_recon_loss': avg_val_recon_loss,
        'val_kl_loss': avg_val_kl_loss,
        'val_weighted_kl_loss': avg_val_weighted_kl_loss,
        'kl_weight': kl_weight
    }
    
    with open('vae_training_history.pkl', 'wb') as file:
        pickle.dump(history, file)
    
    print("VAE training completed. Models and weights saved.")
    
    # Close TensorBoard writer
    writer.close()
    
    return vae, encoder, decoder

if __name__ == "__main__":
    train_vae()
