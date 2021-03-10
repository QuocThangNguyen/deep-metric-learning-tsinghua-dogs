import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
import datetime
import random
from typing import List, Dict, Tuple, Any, Union


def set_random_seed(seed: int) -> None:
    """
    Set random seed for package random, numpy and pytorch
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_current_time() -> str:
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def save_checkpoint(model: nn.Module,
                    config: Dict[str, Any],
                    current_epoch: int,
                    current_iter: int,
                    output_dir: str,
                    mean_average_precision: float = None,
                    ) -> str:

    checkpoint_name: str = f"epoch{current_epoch}-iter{current_iter}"
    if mean_average_precision is not None:
        checkpoint_name += f"-map{mean_average_precision:.2f}"
    checkpoint_name += ".pth"

    checkpoint_path: str = os.path.join(output_dir, checkpoint_name)
    torch.save(
        {
            "config": config,
            "model_state_dict": model.module.state_dict(),
        },
        checkpoint_path
    )
    return checkpoint_path


def log_embeddings_to_tensorboard(loader: DataLoader,
                                  model: nn.Module,
                                  device: torch.device,
                                  writer: SummaryWriter,
                                  tag: str
                                  ) -> None:
    if tag == "train":
        if hasattr(loader.sampler, "sequential_sampling"):
            loader.sampler.sequential_sampling = True
    # Calculating embedding of training set for visualization
    embeddings, labels = get_embeddings_from_dataloader(loader, model, device)
    writer.add_embedding(embeddings, metadata=labels.tolist(), tag=tag)


@torch.no_grad()
def get_embeddings_from_dataloader(loader: DataLoader,
                                   model: nn.Module,
                                   device: torch.device,
                                   return_numpy_array=False,
                                   return_image_paths=False,
                                   ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[np.ndarray, np.ndarray]]:
    model.eval()

    embeddings_ls: List[torch.Tensor] = []
    labels_ls: List[torch.Tensor] = []
    for images_, labels_ in loader:
        images: torch.Tensor = images_.to(device, non_blocking=True)
        labels: torch.Tensor = labels_.to(device, non_blocking=True)
        embeddings: torch.Tensor = model(images)
        embeddings_ls.append(embeddings)
        labels_ls.append(labels)

    embeddings: torch.Tensor = torch.cat(embeddings_ls, dim=0)  # shape: [N x embedding_size]
    labels: torch.Tensor = torch.cat(labels_ls, dim=0)  # shape: [N]

    if return_numpy_array:
        embeddings = embeddings.cpu().numpy()
        labels = labels.cpu().numpy()

    if return_image_paths:
        images_paths: List[str] = []
        for path, _ in loader.dataset.samples:
            images_paths.append(path)
        return (embeddings, labels, images_paths)

    return (embeddings, labels)
