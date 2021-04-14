import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import logging
from typing import Tuple, Dict, Any, Union

from src.losses import TripletMarginLoss, ProxyNCALoss, ProxyAnchorLoss, SoftTripleLoss
from src.utils import save_checkpoint
from src.evaluate import calculate_all_metrics


logger = logging.getLogger(__name__)


def train_one_epoch(model: nn.Module,
                    optimizer: Optimizer,
                    loss_function: Union[TripletMarginLoss, ProxyNCALoss, ProxyAnchorLoss, SoftTripleLoss],
                    train_loader: DataLoader,
                    test_loader: DataLoader,
                    reference_loader: DataLoader,
                    writer: SummaryWriter,
                    device: torch.device,
                    config: Dict[str, Any],
                    checkpoint_dir: str,
                    log_frequency: int,
                    validate_frequency: int,
                    output_dict: Dict[str, Any]
                    ) -> Dict[str, Any]:

    # Increase number of epochs so far
    output_dict["current_epoch"] += 1
    current_epoch: int = output_dict["current_epoch"]

    running_loss: float = 0.0
    running_fraction_hard_triplets: float = 0.0

    for images, labels in train_loader:
        # Increase number of iterations so far
        output_dict["current_iter"] += 1
        current_iter: int = output_dict["current_iter"]

        # Run validation
        if current_iter == 1 or (current_iter % validate_frequency == 0):
            metrics: Dict[str, Any] = calculate_all_metrics(model.module, test_loader, reference_loader, device)
            log_info_metrics(logger, metrics, current_epoch, current_iter)

            # Log all metrics to tensorboard
            for metric_name, value in metrics.items():
                writer.add_scalar(f"test/{metric_name}", value, current_iter)

            # Save checkpoint that has highest MAP
            if metrics['mean_average_precision'] > output_dict["metrics"]["mean_average_precision"]:
                output_dict["metrics"] = metrics
                save_checkpoint(
                    model,
                    config,
                    current_epoch,
                    current_iter,
                    checkpoint_dir,
                    metrics['mean_average_precision'],
                )

        output_batch: Dict[str, Any] = train_one_batch(model, optimizer, loss_function, images, labels, device)
        running_loss += output_batch["loss"]
        running_fraction_hard_triplets += output_batch["fraction_hard_triplets"]

        # Logging to tensorboard
        for metric_name, value in output_batch.items():
            writer.add_scalar(f"train/{metric_name}", value, current_iter)

        # Logging to standard output stream
        if current_iter % log_frequency == 0:
            average_loss: float = running_loss / log_frequency
            average_hard_triplets: float = running_fraction_hard_triplets / log_frequency * 100
            logger.info(
                f"TRAINING\t[{current_epoch}|{current_iter}]\t"
                f"train_loss: {average_loss:.6f}\t"
                f"hard triplets: {average_hard_triplets:.2f}%"
            )
            running_loss = 0.0
            running_fraction_hard_triplets = 0.0

    # Run validation at the final iteration
    if output_dict["current_epoch"] == output_dict["total_epoch"]:
        metrics: Dict[str, Any] = calculate_all_metrics(model.module, test_loader, reference_loader, device)
        log_info_metrics(logger, metrics, current_epoch, current_iter)

        for metric_name, value in metrics.items():
            writer.add_scalar(f"test/{metric_name}", value, current_iter)

        if metrics['mean_average_precision'] > output_dict["metrics"]["mean_average_precision"]:
            output_dict["metrics"] = metrics
            save_checkpoint(
                model,
                config,
                current_epoch,
                current_iter,
                checkpoint_dir,
                metrics['mean_average_precision'],
            )
    return output_dict


def train_one_batch(model: nn.Module,
                    optimizer: Optimizer,
                    loss_function: Union[TripletMarginLoss, ProxyNCALoss, ProxyAnchorLoss, SoftTripleLoss],
                    images: torch.Tensor,
                    labels: torch.Tensor,
                    device: torch.device,
                    ) -> Tuple[float, float]:
    model.train()
    optimizer.zero_grad()

    images: torch.Tensor = images.to(device, non_blocking=True)
    labels: torch.Tensor = labels.to(device, non_blocking=True)

    embeddings: torch.Tensor = model(images)
    loss, fraction_hard_triplets = loss_function(embeddings, labels)

    loss.backward()
    optimizer.step()

    return {
        "loss": loss.item(),
        "fraction_hard_triplets": float(fraction_hard_triplets)
    }


def log_info_metrics(logger, metrics: Dict[str, float], current_epoch: int, current_iter: int) -> None:
    """
    Print all metrics to stdout
    """
    logger.info("*" * 130)
    logger.info(
        f"VALIDATING\t[{current_epoch}|{current_iter}]\t"
        f"MAP: {metrics['mean_average_precision']:.2f}%\t"
        f"AP@1: {metrics['average_precision_at_1']:.2f}%\t"
        f"AP@5: {metrics['average_precision_at_5']:.2f}%\t"
        f"Top-1: {metrics['top_1_accuracy']:.2f}%\t"
        f"Top-5: {metrics['top_5_accuracy']:.2f}%\t"
        f"NMI: {metrics['normalized_mutual_information']:.2f}\t"
    )
    logger.info("*" * 130)
