"""
Script to run the simplified VAE training.
"""
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the simplified training function
from simple_train_vae import train_vae_simple

if __name__ == "__main__":
    print("Starting simplified VAE training...")
    train_vae_simple()
