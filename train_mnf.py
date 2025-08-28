#!/usr/bin/env python3

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchinfo
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from core.model import Moses
from core.utils import data_loader, metrics


def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("mnf_training.log")],
    )
    return logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Training Script for Marginal Normalizing Flows",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training parameters
    training_group = parser.add_argument_group("Training Parameters")
    training_group.add_argument(
        "--epochs", "-e", type=int, default=1000, help="Maximum number of epochs"
    )
    training_group.add_argument(
        "--batch-size", "-bs", type=int, default=64, help="Training batch size"
    )
    training_group.add_argument(
        "--val-batch-size", "-vbs", type=int, default=50, help="Validation batch size"
    )
    training_group.add_argument(
        "--learn-rate", "-lr", type=float, default=1e-3, help="Learning rate"
    )
    training_group.add_argument(
        "--betas",
        "-b",
        nargs=2,
        type=float,
        default=[0.9, 0.999],
        help="Adam optimizer betas",
    )
    training_group.add_argument(
        "--weight-decay",
        "-wd",
        type=float,
        default=1e-3,
        help="Weight decay for regularization",
    )
    training_group.add_argument(
        "--seed", "-s", type=int, default=42, help="Random seed for reproducibility"
    )

    # Model parameters
    model_group = parser.add_argument_group("Model Parameters")
    model_group.add_argument(
        "--flayers", "-fl", type=int, default=1, help="Number of flow layers"
    )
    model_group.add_argument(
        "--n-gaussians",
        "-ng",
        type=int,
        default=3,
        help="Number of Gaussian components",
    )
    model_group.add_argument(
        "--use-cov",
        type=int,
        default=1,
        choices=[0, 1],
        help="Use covariance matrix for base Gaussian (1=True, 0=False)",
    )
    model_group.add_argument(
        "--use-activation",
        type=int,
        default=0,
        choices=[0, 1],
        help="Use activation between splines (1=True, 0=False)",
    )
    model_group.add_argument(
        "--n-heads", "-nh", type=int, default=4, help="Number of attention heads"
    )
    model_group.add_argument(
        "--n-hiddens", type=int, default=128, help="Number of hidden dimensions"
    )

    # Dataset parameters
    data_group = parser.add_argument_group("Dataset Parameters")
    data_group.add_argument(
        "--dataset",
        "-dset",
        type=str,
        default="ushcn",
        choices=["ushcn", "mimiciii", "mimiciv", "physionet2012"],
        help="Dataset to use",
    )
    data_group.add_argument(
        "--forc-time", "-ft", type=int, default=0, help="Forecast horizon in hours"
    )
    data_group.add_argument(
        "--cond-time", "-ct", type=int, default=36, help="Conditioning range in hours"
    )
    data_group.add_argument(
        "--nfolds",
        "-nf",
        type=int,
        default=5,
        help="Number of folds for cross-validation",
    )
    data_group.add_argument(
        "--fold", "-f", type=int, default=0, help="Current fold number"
    )

    # Additional parameters
    misc_group = parser.add_argument_group("Miscellaneous")
    misc_group.add_argument(
        "--output-dir",
        type=str,
        default="saved_models",
        help="Directory to save models",
    )
    misc_group.add_argument(
        "--patience", type=int, default=30, help="Early stopping patience"
    )
    misc_group.add_argument(
        "--scheduler-patience", type=int, default=10, help="Scheduler patience"
    )

    return parser.parse_args()


def setup_environment(seed: int) -> torch.device:
    """Setup environment and random seeds."""
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # CUDA optimizations
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Enable anomaly detection for debugging
    torch.autograd.set_detect_anomaly(True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def load_dataset(args: argparse.Namespace):
    """Load the specified dataset."""
    dataset_config = {
        "normalize_time": True,
        "condition_time": args.cond_time,
        "forecast_horizon": args.forc_time,
        "num_folds": args.nfolds,
    }

    if args.dataset == "ushcn":
        from tsdm.tasks import USHCN_DeBrouwer2019

        return USHCN_DeBrouwer2019(**dataset_config)
    elif args.dataset == "mimiciii":
        from tsdm.tasks.mimic_iii_debrouwer2019 import MIMIC_III_DeBrouwer2019

        return MIMIC_III_DeBrouwer2019(**dataset_config)
    elif args.dataset == "mimiciv":
        from tsdm.tasks.mimic_iv_bilos2021 import MIMIC_IV_Bilos2021

        return MIMIC_IV_Bilos2021(**dataset_config)
    elif args.dataset == "physionet2012":
        from tsdm.tasks.physionet2012 import Physionet2012

        return Physionet2012(**dataset_config)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


def create_dataloaders(task, args: argparse.Namespace):
    """Create train, validation, and test dataloaders."""
    train_loader, val_loader, test_loader = data_loader.data_loaders(task, args)
    return train_loader, val_loader, test_loader


class Trainer:
    """Elegant trainer class for Marginal Normalizing Flows model."""

    def __init__(
        self,
        model: nn.Module,
        args,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        logger: logging.Logger,
    ):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logger
        self.best_val_loss = float("inf")
        self.early_stop_counter = 0
        self.compute_loss = metrics.compute_losses()

    def train_epoch(self, train_loader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for batch in train_loader:
            # Move batch to device and unpack
            obs, mobs, xq, mq, y = (tensor.to(self.device) for tensor in batch)

            # Skip empty batches
            if xq.shape[1] == 0:
                continue

            # Forward pass
            self.optimizer.zero_grad()
            z, ldj, mw = self.model(obs, mobs, xq, mq, y)

            # Compute loss
            n_samples = mq.sum(-1).bool().sum()
            loss = self.compute_loss.nll(z, mq, ldj, mw)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Accumulate statistics
            total_loss += loss.item() * n_samples.item()
            total_samples += n_samples.item()

        return total_loss / total_samples if total_samples > 0 else 0.0

    def evaluate(self, data_loader) -> float:
        """Evaluate model on given data loader."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device and unpack
                obs, mobs, xq, mq, y = (tensor.to(self.device) for tensor in batch)

                # Skip empty batches
                if xq.shape[1] == 0:
                    continue

                # Forward pass
                z, ldj, mw = self.model(obs, mobs, xq, mq, y)
                n_samples = mq.sum(-1).bool().sum()
                loss = self.compute_loss.nll(z, mq, ldj, mw)

                # Accumulate statistics
                total_loss += loss.item() * n_samples.item()
                total_samples += n_samples.item()

        return total_loss / total_samples if total_samples > 0 else 0.0

    def save_checkpoint(self, filepath: str, epoch: int, args: argparse.Namespace):
        """Save model checkpoint."""
        torch.save(
            {
                "args": args,
                "epoch": epoch,
                "state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": self.best_val_loss,
            },
            filepath,
        )

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        return checkpoint


def main():
    """Main training function."""
    # Setup
    args = parse_arguments()
    logger = setup_logging()
    device = setup_environment(args.seed)

    logger.info("Arguments: %s", args)
    experiment_id = int(time.time() * 1000) % 10000000
    logger.info("Starting training with experiment ID: %d", experiment_id)
    logger.info("Using device: %s", device)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load dataset and create dataloaders
    task = load_dataset(args)
    train_loader, valid_loader, test_loader = create_dataloaders(task, args)

    # Initialize model
    model_config = {
        "n_inputs": task.dataset.shape[-1],
        "n_gaussians": args.n_gaussians,
        "use_cov": bool(args.use_cov),
        "use_activation": bool(args.use_activation),
        "n_hiddens": args.n_hiddens,
        "n_flayers": args.flayers,
        "n_heads": args.n_heads,
        "device": device,
    }

    model = MarginalNormalizingFlows(**model_config).to(device)
    model.zero_grad(set_to_none=True)

    # Print model summary
    logger.info(
        "Model initialized with %d parameters",
        sum(p.numel() for p in model.parameters()),
    )

    # Initialize optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.learn_rate,
        betas=args.betas,
        weight_decay=args.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=args.scheduler_patience,
        factor=0.5,
        min_lr=1e-5,
        verbose=True,
    )

    # Initialize trainer
    trainer = Trainer(model, args, optimizer, scheduler, device, logger)

    # Generate model path
    model_path = output_dir / f"mnf_{args.dataset}_fold{args.fold}_{experiment_id}.pt"

    # Training loop
    logger.info("Starting training loop...")
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # Train
        train_loss = trainer.train_epoch(train_loader)

        # Validate
        val_loss = trainer.evaluate(valid_loader)

        epoch_time = time.time() - start_time

        logger.info(
            "Epoch %3d/%d | Train Loss: %.6f | Val Loss: %.6f | Time: %.2fs",
            epoch,
            args.epochs,
            train_loss,
            val_loss,
            epoch_time,
        )

        # Save best model and check early stopping
        if val_loss < trainer.best_val_loss:
            trainer.best_val_loss = val_loss
            trainer.save_checkpoint(str(model_path), epoch, args)
            trainer.early_stop_counter = 0
            logger.info("New best model saved with val_loss: %.6f", val_loss)
        else:
            trainer.early_stop_counter += 1

        # Early stopping
        if trainer.early_stop_counter >= args.patience:
            logger.info(
                "Early stopping after %d epochs without improvement",
                args.patience,
            )
            break

        # Update scheduler
        scheduler.step(val_loss)

    # Final evaluation on test set
    logger.info("Loading best model for final evaluation...")
    trainer.load_checkpoint(str(model_path))

    start_time = time.time()
    test_loss = trainer.evaluate(test_loader)
    eval_time = time.time() - start_time

    logger.info("Final Results:")
    logger.info("Best Val Loss: %.6f", trainer.best_val_loss)
    logger.info("Test Loss: %.6f", test_loss)
    logger.info("Evaluation Time: %.2fs", eval_time)

    return {
        "best_val_loss": trainer.best_val_loss,
        "test_loss": test_loss,
        "experiment_id": experiment_id,
    }


if __name__ == "__main__":
    print(" ".join(sys.argv))
    main()
    main()
    main()
    main()
    main()
    main()
    main()
