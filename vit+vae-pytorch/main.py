"""
Main entry point for training and inference in PyTorch implementation.
"""
import argparse
import torch
import os

def main():
    """Main function."""
    # Update device configuration based on availability
    import config.model_config as cfg
    cfg.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='VAE-ViT Hybrid Model for Plant Disease Classification')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train VAE command
    train_vae_parser = subparsers.add_parser('train-vae', help='Train the VAE model')
    train_vae_parser.add_argument('--epochs', type=int, default=cfg.VAE_EPOCHS, help='Number of epochs to train')
    train_vae_parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    
    # Train hybrid model command
    train_hybrid_parser = subparsers.add_parser('train-hybrid', help='Train the hybrid model')
    train_hybrid_parser.add_argument('--epochs', type=int, default=cfg.HYBRID_EPOCHS, help='Number of epochs to train')
    train_hybrid_parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    
    # Inference command
    inference_parser = subparsers.add_parser('inference', help='Run inference on a single image')
    inference_parser.add_argument('--image', type=str, required=True, help='Path to the image file')
    inference_parser.add_argument('--visualize', action='store_true', help='Visualize the prediction')
    inference_parser.add_argument('--output', type=str, help='Path to save the visualization')
    
    # Batch inference command
    batch_parser = subparsers.add_parser('batch', help='Run inference on multiple images')
    batch_parser.add_argument('--input_dir', type=str, required=True, help='Directory containing images')
    batch_parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results')
    batch_parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    batch_parser.add_argument('--visualize', action='store_true', help='Visualize the predictions')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs('saved_models/vae', exist_ok=True)
    os.makedirs('saved_models/hybrid', exist_ok=True)
    os.makedirs('checkpoints/vae', exist_ok=True)
    os.makedirs('checkpoints/hybrid', exist_ok=True)
    
    # Handle commands
    if args.command == "train-vae":
        print(f"Training VAE for {args.epochs} epochs...")
        from training.train_vae import train_vae
        # Update epochs in config
        cfg.VAE_EPOCHS = args.epochs
        # Run training
        train_vae(epochs=args.epochs, resume=args.resume)
        
    elif args.command == "train-hybrid":
        print(f"Training hybrid model for {args.epochs} epochs...")
        from training.train_hybrid import train_hybrid_model
        # Update epochs in config
        cfg.HYBRID_EPOCHS = args.epochs
        # Run training
        train_hybrid_model(epochs=args.epochs, resume=args.resume)
        
    elif args.command == "inference":
        print(f"Running inference on image: {args.image}")
        from inference.inference import main as inference_main
        # Run inference
        import sys
        sys.argv = ['inference.py', '--image', args.image]
        if args.visualize:
            sys.argv.append('--visualize')
        if args.output:
            sys.argv.extend(['--output', args.output])
        inference_main()
        
    elif args.command == "batch":
        print(f"Running batch inference on images in: {args.input_dir}")
        from inference.batch_inference import main as batch_main
        # Run batch inference
        import sys
        sys.argv = ['batch_inference.py', '--input_dir', args.input_dir, '--output_dir', args.output_dir, '--batch_size', str(args.batch_size)]
        if args.visualize:
            sys.argv.append('--visualize')
        batch_main()
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
