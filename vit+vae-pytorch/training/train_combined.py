"""
Training script for the combined VAE-ViT model in PyTorch.
Both components are trained simultaneously.
"""
import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import pickle
from tqdm import tqdm

from models.vae import build_encoder, build_decoder
from models.vit import build_vit_classifier
from models.combined_model import build_combined_model
from data.data_loader import initialize_data_loaders
from utils.model_utils import save_checkpoint, load_checkpoint, get_device
from config.model_config import (
    LEARNING_RATE, HYBRID_EPOCHS, BATCH_SIZE,
    VAE_WEIGHTS_PATH, HYBRID_WEIGHTS_PATH, HYBRID_CHECKPOINTS_PATH
)

def train_combined_model(epochs=HYBRID_EPOCHS, resume=False):
    """
    Train the combined VAE-ViT model where both components are trained simultaneously.
    
    Args:
        epochs (int): Number of epochs to train
        resume (bool): Whether to resume training from a checkpoint
        
    Returns:
        CombinedModel: Trained combined model
    """
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Initialize data loaders
    print("Initializing data loaders...")
    train_loader, val_loader, _ = initialize_data_loaders(verbose=True)
    
    # Build the models
    print("Building models...")
    encoder = build_encoder()
    decoder = build_decoder()
    vit_classifier = build_vit_classifier()
    
    # Get the size of the training dataset
    train_data_size = len(train_loader.dataset)
    print(f"Training dataset size: {train_data_size} images")
    
    # Build the combined model
    combined_model = build_combined_model(
        encoder, decoder, vit_classifier, 
        batch_size=BATCH_SIZE,
        train_data_size=train_data_size
    )
    
    # Move model to device
    combined_model = combined_model.to(device)
    
    # Calculate and print parameter counts
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    total_params = count_parameters(combined_model)
    vae_params = count_parameters(encoder) + count_parameters(decoder)
    vit_params = count_parameters(vit_classifier)
    
    print(f"\nModel Parameter Counts:")
    print(f"Total parameters: {total_params:,}")
    print(f"VAE parameters (encoder+decoder): {vae_params:,}")
    print(f"ViT parameters: {vit_params:,}")
    print(f"VAE percentage: {(vae_params/total_params)*100:.2f}%")
    print(f"ViT percentage: {(vit_params/total_params)*100:.2f}%\n")
    
    # Create optimizer for all parameters
    optimizer = optim.Adam(combined_model.parameters(), lr=LEARNING_RATE)
    
    # Create directories for saving weights
    os.makedirs(VAE_WEIGHTS_PATH, exist_ok=True)
    os.makedirs(HYBRID_WEIGHTS_PATH, exist_ok=True)
    os.makedirs(HYBRID_CHECKPOINTS_PATH, exist_ok=True)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(HYBRID_CHECKPOINTS_PATH, "combined_logs"))
    
    # Starting epoch and best metrics
    start_epoch = 0
    best_loss = float('inf')
    best_accuracy = 0.0
    
    # Resume training if requested
    if resume:
        checkpoint_path = os.path.join(HYBRID_CHECKPOINTS_PATH, "combined_latest.pt")
        combined_model, optimizer, start_epoch, best_loss = load_checkpoint(combined_model, optimizer, checkpoint_path)
    
    # Training loop
    print(f"Training combined model for {epochs} epochs...")
    for epoch in range(start_epoch, epochs):
        # Training phase
        combined_model.train()
        train_loss = 0
        train_recon_loss = 0
        train_class_loss = 0
        train_kl_loss = 0
        train_weighted_kl_loss = 0
        train_correct = 0
        train_total = 0
        
        # Update current epoch in combined model for KL annealing
        combined_model.current_epoch = epoch
        
        # Progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (data, targets) in enumerate(progress_bar):
            # Move data to device
            data = data.to(device)
            targets = targets.to(device)
            
            # Update current iteration in combined model for KL annealing
            combined_model.current_iteration = epoch * len(train_loader) + batch_idx
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed, class_outputs, mean, log_var, z = combined_model(data)
            
            # Calculate loss
            loss, recon_loss, class_loss, kl_loss, weighted_kl, kl_weight = combined_model.loss_function(
                reconstructed, data, class_outputs, targets, mean, log_var
            )
            # Backward pass ( calculate gradients)
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_class_loss += class_loss.item()
            train_kl_loss += kl_loss.item()
            train_weighted_kl_loss += weighted_kl.item()
            
            # Calculate accuracy
            _, predicted = class_outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            train_accuracy = 100.0 * train_correct / train_total
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'recon': f"{recon_loss.item():.4f}",
                'class': f"{class_loss.item():.4f}",
                'kl': f"{kl_loss.item():.4f}",
                'acc': f"{train_accuracy:.2f}%"
            })
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_train_recon_loss = train_recon_loss / len(train_loader)
        avg_train_class_loss = train_class_loss / len(train_loader)
        avg_train_kl_loss = train_kl_loss / len(train_loader)
        avg_train_weighted_kl_loss = train_weighted_kl_loss / len(train_loader)
        avg_train_accuracy = 100.0 * train_correct / train_total
        
        # Validation phase
        combined_model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_class_loss = 0
        val_kl_loss = 0
        val_weighted_kl_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                # Move data to device
                data = data.to(device)
                targets = targets.to(device)
                
                # Forward pass
                reconstructed, class_outputs, mean, log_var, z = combined_model(data)
                
                # Calculate loss
                loss, recon_loss, class_loss, kl_loss, weighted_kl, _ = combined_model.loss_function(
                    reconstructed, data, class_outputs, targets, mean, log_var
                )
                
                # Update metrics
                val_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_class_loss += class_loss.item()
                val_kl_loss += kl_loss.item()
                val_weighted_kl_loss += weighted_kl.item()
                
                # Calculate accuracy
                _, predicted = class_outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        # Calculate average validation metrics
        avg_val_loss = val_loss / len(val_loader)
        avg_val_recon_loss = val_recon_loss / len(val_loader)
        avg_val_class_loss = val_class_loss / len(val_loader)
        avg_val_kl_loss = val_kl_loss / len(val_loader)
        avg_val_weighted_kl_loss = val_weighted_kl_loss / len(val_loader)
        avg_val_accuracy = 100.0 * val_correct / val_total
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{epochs} Summary:")
        print(f"Train Loss: {avg_train_loss:.4f}, Recon: {avg_train_recon_loss:.4f}, "
              f"Class: {avg_train_class_loss:.4f}, KL: {avg_train_kl_loss:.4f}, "
              f"Weighted KL: {avg_train_weighted_kl_loss:.4f}, Accuracy: {avg_train_accuracy:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f}, Recon: {avg_val_recon_loss:.4f}, "
              f"Class: {avg_val_class_loss:.4f}, KL: {avg_val_kl_loss:.4f}, "
              f"Weighted KL: {avg_val_weighted_kl_loss:.4f}, Accuracy: {avg_val_accuracy:.2f}%")
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Reconstruction_Loss/train', avg_train_recon_loss, epoch)
        writer.add_scalar('Reconstruction_Loss/val', avg_val_recon_loss, epoch)
        writer.add_scalar('Classification_Loss/train', avg_train_class_loss, epoch)
        writer.add_scalar('Classification_Loss/val', avg_val_class_loss, epoch)
        writer.add_scalar('KL_Loss/train', avg_train_kl_loss, epoch)
        writer.add_scalar('KL_Loss/val', avg_val_kl_loss, epoch)
        writer.add_scalar('Weighted_KL_Loss/train', avg_train_weighted_kl_loss, epoch)
        writer.add_scalar('Weighted_KL_Loss/val', avg_val_weighted_kl_loss, epoch)
        writer.add_scalar('Accuracy/train', avg_train_accuracy, epoch)
        writer.add_scalar('Accuracy/val', avg_val_accuracy, epoch)
        
        # Save checkpoint
        checkpoint = {
            'model_state_dict': combined_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch + 1,
            'loss': avg_val_loss
        }
        
        # Save latest checkpoint
        latest_checkpoint_path = os.path.join(HYBRID_CHECKPOINTS_PATH, "combined_latest.pt")
        torch.save(checkpoint, latest_checkpoint_path)
        
        # Save best model based on validation loss
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_checkpoint_path = os.path.join(HYBRID_CHECKPOINTS_PATH, "combined_best_loss.pt")
            torch.save(checkpoint, best_checkpoint_path)
            print(f"New best validation loss: {best_loss:.4f}")
        
        # Save best model based on validation accuracy
        if avg_val_accuracy > best_accuracy:
            best_accuracy = avg_val_accuracy
            best_acc_checkpoint_path = os.path.join(HYBRID_CHECKPOINTS_PATH, "combined_best_accuracy.pt")
            torch.save(checkpoint, best_acc_checkpoint_path)
            print(f"New best validation accuracy: {best_accuracy:.2f}%")
    
    # Save the final model components separately
    encoder_path = os.path.join(VAE_WEIGHTS_PATH, "encoder.pt")
    decoder_path = os.path.join(VAE_WEIGHTS_PATH, "decoder.pt")
    vit_path = os.path.join(HYBRID_WEIGHTS_PATH, "vit_classifier.pt")
    combined_path = os.path.join(HYBRID_WEIGHTS_PATH, "combined_model.pt")
    
    torch.save(combined_model.encoder.state_dict(), encoder_path)
    torch.save(combined_model.decoder.state_dict(), decoder_path)
    torch.save(combined_model.vit_classifier.state_dict(), vit_path)
    torch.save(combined_model.state_dict(), combined_path)
    
    # Save the training history
    history = {
        'train_loss': avg_train_loss,
        'train_recon_loss': avg_train_recon_loss,
        'train_class_loss': avg_train_class_loss,
        'train_kl_loss': avg_train_kl_loss,
        'train_accuracy': avg_train_accuracy,
        'val_loss': avg_val_loss,
        'val_recon_loss': avg_val_recon_loss,
        'val_class_loss': avg_val_class_loss,
        'val_kl_loss': avg_val_kl_loss,
        'val_accuracy': avg_val_accuracy,
        'best_loss': best_loss,
        'best_accuracy': best_accuracy
    }
    
    with open('combined_training_history.pkl', 'wb') as file:
        pickle.dump(history, file)
    
    print("Combined model training completed. Model and weights saved.")
    
    # Close TensorBoard writer
    writer.close()
    
    return combined_model

if __name__ == "__main__":
    train_combined_model()