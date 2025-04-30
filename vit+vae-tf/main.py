"""
Main entry point for the VAE-ViT hybrid model project.
"""
import argparse
import os

def main():
    """Main function to handle command line arguments and run the appropriate script."""
    parser = argparse.ArgumentParser(
        description="VAE-ViT Hybrid Model for Plant Disease Classification",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train VAE command
    train_vae_parser = subparsers.add_parser("train-vae", help="Train the VAE model")
    train_vae_parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    
    # Train hybrid command
    train_hybrid_parser = subparsers.add_parser("train-hybrid", help="Train the hybrid model")
    train_hybrid_parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    
    # Inference command
    inference_parser = subparsers.add_parser("inference", help="Run inference on a single image")
    inference_parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    inference_parser.add_argument("--save_dir", type=str, default="inference_results", 
                                help="Directory to save results")
    
    # Batch inference command
    batch_parser = subparsers.add_parser("batch", help="Run inference on multiple images")
    batch_parser.add_argument("--input_dir", type=str, required=True, help="Directory containing images")
    batch_parser.add_argument("--output_dir", type=str, default="batch_results", 
                             help="Directory to save results")
    batch_parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    batch_parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle commands
    if args.command == "train-vae":
        print(f"Training VAE for {args.epochs} epochs...")
        from training.train_vae import train_vae
        # Update epochs in config
        import config.model_config as cfg
        cfg.VAE_EPOCHS = args.epochs
        # Run training
        train_vae()
        
    elif args.command == "train-hybrid":
        print(f"Training hybrid model for {args.epochs} epochs...")
        from training.train_hybrid import train_hybrid_model
        # Update epochs in config
        import config.model_config as cfg
        cfg.HYBRID_EPOCHS = args.epochs
        # Run training
        train_hybrid_model()
        
    elif args.command == "inference":
        print(f"Running inference on image: {args.image}")
        from inference.inference import main as inference_main
        # Set arguments in sys.argv
        import sys
        sys.argv = [sys.argv[0], "--image", args.image, "--save_dir", args.save_dir]
        # Run inference
        inference_main()
        
    elif args.command == "batch":
        print(f"Running batch inference on directory: {args.input_dir}")
        from inference.batch_inference import main as batch_main
        # Set arguments in sys.argv
        import sys
        sys_args = [sys.argv[0], "--input_dir", args.input_dir, "--output_dir", args.output_dir, 
                   "--batch_size", str(args.batch_size)]
        if args.visualize:
            sys_args.append("--visualize")
        sys.argv = sys_args
        # Run batch inference
        batch_main()
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
