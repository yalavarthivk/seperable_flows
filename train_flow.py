"""
Main Training Script

This is the entry point for training neural flow models.
Usage: python train.py --dataset <dataset_name> --model <model_type> [options]
"""

import argparse
import random
import sys
import time
from random import SystemRandom

import numpy as np
import torch

from nfe.config import EXPERIMENT_ID_RANGE, create_argument_parser, setup_random_seeds
from nfe.data_utils_wrapper import create_data_loaders, load_dataset
from nfe.experiments.gru_ode_bayes.experiment import GOB
from nfe.training import train_model


def main():
    """Main execution function."""
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    # Generate unique experiment ID
    experiment_id = int(SystemRandom().random() * EXPERIMENT_ID_RANGE)

    # Display configuration
    print("=" * 80)
    print("Neural Flow Training")
    print("=" * 80)
    print(f"Command: {' '.join(sys.argv)}")
    print(f"Experiment ID: {experiment_id}")
    print(f"Arguments: {args}")
    print("=" * 80)

    # Setup
    setup_random_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.data = args.dataset

    print(f"Device: {device}")

    # Load data
    print("\nLoading dataset...")
    task = load_dataset(args)
    dataloaders = create_data_loaders(task, args)
    print(f"Dataset loaded: {args.dataset}")

    # Initialize model
    print("\nInitializing model...")
    sample_batch = next(iter(dataloaders["train"]))
    input_size = sample_batch["M"].shape[-1]

    model = GOB().get_model(args, input_size)
    model.to(device)

    # Setup optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=args.betas
    )

    # Train model
    train_model(model, dataloaders, optimizer, args, device, experiment_id)


if __name__ == "__main__":
    main()
