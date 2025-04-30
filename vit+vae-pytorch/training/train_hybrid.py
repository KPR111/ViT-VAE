"""
Training script for the hybrid VAE-ViT model in PyTorch.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import pickle
from tqdm import tqdm

from models.vae import build_encoder
from models.vit import build_vit_classifier
from models.hybrid import build_hybrid_model
from data.data_loader import initialize_data_loaders
from utils.model_utils import save_checkpoint, load_checkpoint, load_vae_encoder, verify_vae_weights, get_device
from config.model_config import (
    LEARNING_RATE, HYBRID_EPOCHS,
    HYBRID_WEIGHTS_PATH, HYBRID_CHECKPOINTS_PATH
)

def train_hybrid_model(epochs=HYBRID_EPOCHS, resume=False):
    """
    Train the hybrid VAE-ViT model.
    
    Args:
        epochs (int): Number of epochs to train
        resume (bool): Whether to resume training from a checkpoint
        
    Returns:
        HybridModel: Trained hybrid model
    """
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Verify VAE weights
    if not verify_vae_weights():
        print("VAE weights not found. Please train the VAE first.")
        return None
    
    # Initialize data loaders
    print("Initializing data loaders...")
    train_loader, val_loader, _ = initialize_data_loaders(verbose=True)
    
    # Load pre-trained VAE encoder
    print("Loading pre-trained VAE encoder...")
    encoder = load_vae_encoder()
    if encoder is None:
        print("Failed to load VAE encoder. Please train the VAE first.")
        return None
    
    # Build ViT classifier
    print("Building ViT classifier...")
    vit_classifier = build_vit_classifier()
    vit_classifier = vit_classifier.to(device)
    
    # Freeze VAE Encoder weights
    for param in encoder.parameters():
        param.requires_grad = False
    print("VAE Encoder layers frozen")
    
    # Build and compile hybrid model
    print("Building hybrid model...")
    hybrid_model = build_hybrid_model(encoder, vit_classifier)
    hybrid_model = hybrid_model.to(device)
    
    # Create optimizer (only for ViT classifier parameters)
    optimizer = optim.Adam(vit_classifier.parameters(), lr=LEARNING_RATE)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create directories for saving weights
    os.makedirs(HYBRID_WEIGHTS_PATH, exist_ok=True)
    os.makedirs(HYBRID_CHECKPOINTS_PATH, exist_ok=True)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(HYBRID_CHECKPOINTS_PATH, "logs"))
    
    # Starting epoch and best metrics
    start_epoch = 0
    best_loss = float('inf')
    best_accuracy = 0.0
    
    # Resume training if requested
    if resume:
        checkpoint_path = os.path.join(HYBRID_CHECKPOINTS_PATH, "hybrid_latest.pt")
        hybrid_model, optimizer, start_epoch, best_loss = load_checkpoint(hybrid_model, optimizer, checkpoint_path)
    
    # Training loop
    print(f"Training hybrid model for {epochs} epochs...")
    for epoch in range(start_epoch, epochs):
        # Training phase
        hybrid_model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        # Progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (data, targets) in enumerate(progress_bar):
            # Move data to device
            data = data.to(device)
            targets = targets.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = hybrid_model(data)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            # Calculate accuracy
            train_accuracy = 100.0 * train_correct / train_total
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{train_accuracy:.2f}%"
            })
        
        # Calculate average loss and accuracy
        avg_train_loss = train_loss / len(train_loader)
        avg_train_accuracy = 100.0 * train_correct / train_total
        
        # Validation phase
        hybrid_model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                # Move data to device
                data = data.to(device)
                targets = targets.to(device)
                
                # Forward pass
                outputs = hybrid_model(data)
                
                # Calculate loss
                loss = criterion(outputs, targets)
                
                # Update metrics
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        # Calculate average loss and accuracy
        avg_val_loss = val_loss / len(val_loader)
        avg_val_accuracy = 100.0 * val_correct / val_total
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_accuracy:.2f}%, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_accuracy:.2f}%")
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/train', avg_train_accuracy, epoch)
        writer.add_scalar('Accuracy/val', avg_val_accuracy, epoch)
        
        # Save checkpoint for every epoch
        checkpoint_path = os.path.join(HYBRID_CHECKPOINTS_PATH, f"hybrid_model_epoch_{epoch+1:02d}.pt")
        save_checkpoint(hybrid_model, optimizer, epoch+1, avg_val_loss, checkpoint_path)
        
        # Save latest checkpoint (for resuming)
        latest_checkpoint_path = os.path.join(HYBRID_CHECKPOINTS_PATH, "hybrid_latest.pt")
        save_checkpoint(hybrid_model, optimizer, epoch+1, avg_val_loss, latest_checkpoint_path)
        
        # Save best model based on validation loss
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_loss_checkpoint_path = os.path.join(HYBRID_CHECKPOINTS_PATH, "hybrid_best_loss.pt")
            save_checkpoint(hybrid_model, optimizer, epoch+1, avg_val_loss, best_loss_checkpoint_path)
            print(f"New best model saved with validation loss: {best_loss:.4f}")
        
        # Save best model based on validation accuracy
        if avg_val_accuracy > best_accuracy:
            best_accuracy = avg_val_accuracy
            best_acc_checkpoint_path = os.path.join(HYBRID_CHECKPOINTS_PATH, "hybrid_best_acc.pt")
            save_checkpoint(hybrid_model, optimizer, epoch+1, avg_val_loss, best_acc_checkpoint_path)
            print(f"New best model saved with validation accuracy: {best_accuracy:.2f}%")
    
    # Save the final model
    hybrid_path = os.path.join(HYBRID_WEIGHTS_PATH, "hybrid_model.pt")
    torch.save(hybrid_model.state_dict(), hybrid_path)
    
    # Save the training history
    history = {
        'train_loss': avg_train_loss,
        'train_accuracy': avg_train_accuracy,
        'val_loss': avg_val_loss,
        'val_accuracy': avg_val_accuracy,
        'best_loss': best_loss,
        'best_accuracy': best_accuracy
    }
    
    with open('hybrid_training_history.pkl', 'wb') as file:
        pickle.dump(history, file)
    
    print("Hybrid model training completed. Model and weights saved.")
    
    # Close TensorBoard writer
    writer.close()
    
    return hybrid_model

if __name__ == "__main__":
    train_hybrid_model()
