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

    # Combined model is the only training option now

    # Train combined model command (which is our hybrid model)
    train_combined_parser = subparsers.add_parser('train-combined', help='Train the combined model (VAE and ViT simultaneously)')
    train_combined_parser.add_argument('--epochs', type=int, default=cfg.HYBRID_EPOCHS, help='Number of epochs to train')
    train_combined_parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')

    # Test command
    test_parser = subparsers.add_parser('test', help='Test the combined model on the test dataset')
    test_parser.add_argument('--test_dir', type=str, default=cfg.TEST_DIR, help='Path to the test directory')
    test_parser.add_argument('--output_dir', type=str, default='test_results', help='Directory to save test results')

    # Check test directory command
    check_test_parser = subparsers.add_parser('check-test-dir', help='Check the test directory and print information about its contents')
    check_test_parser.add_argument('--test_dir', type=str, default=cfg.TEST_DIR, help='Path to the test directory')

    # Inference command
    inference_parser = subparsers.add_parser('inference', help='Run inference on a single image')
    inference_parser.add_argument('--image', type=str, required=True, help='Path to the image file')
    inference_parser.add_argument('--visualize', action='store_true', help='Visualize the prediction')
    inference_parser.add_argument('--output', type=str, help='Path to save the visualization')
    inference_parser.add_argument('--model', type=str, default='combined', choices=['combined'],
                                 help='Model type to use for inference (combined model)')

    # Batch inference command
    batch_parser = subparsers.add_parser('batch', help='Run inference on multiple images')
    batch_parser.add_argument('--input_dir', type=str, required=True, help='Directory containing images')
    batch_parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results')
    batch_parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    batch_parser.add_argument('--visualize', action='store_true', help='Visualize the predictions')
    batch_parser.add_argument('--model', type=str, default='combined', choices=['combined'],
                             help='Model type to use for inference (combined model)')

    args = parser.parse_args()

    # Create necessary directories
    os.makedirs('saved_models/vae', exist_ok=True)
    os.makedirs('saved_models/hybrid', exist_ok=True)
    os.makedirs('checkpoints/vae', exist_ok=True)
    os.makedirs('checkpoints/hybrid', exist_ok=True)

    # Handle commands
    if args.command == "train-combined":
        print(f"Training combined model for {args.epochs} epochs...")
        from training.train_combined import train_combined_model
        # Update epochs in config
        cfg.HYBRID_EPOCHS = args.epochs
        # Run training
        train_combined_model(epochs=args.epochs, resume=args.resume)

    elif args.command == "check-test-dir":
        print(f"Checking test directory: {args.test_dir}")
        from data.test_data_loader import check_test_directory
        # Check test directory
        check_test_directory(test_dir=args.test_dir)

    elif args.command == "test":
        print(f"Testing combined model on test dataset...")
        from training.test_combined import test_combined_model
        # Run testing
        test_combined_model(test_dir=args.test_dir, output_dir=args.output_dir)

    elif args.command == "inference":
        print(f"Running inference on image: {args.image}")
        from inference.inference import main as inference_main
        # Run inference
        import sys
        sys.argv = ['inference.py', '--image', args.image, '--model', args.model]
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
        sys.argv = ['batch_inference.py', '--input_dir', args.input_dir, '--output_dir', args.output_dir,
                    '--batch_size', str(args.batch_size), '--model', args.model]
        if args.visualize:
            sys.argv.append('--visualize')
        batch_main()

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
