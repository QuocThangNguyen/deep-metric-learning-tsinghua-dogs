import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from typing import Tuple, Dict
from src.utils import get_embeddings_from_dataloader


def calculate_all_metrics(model: nn.Module,
                          test_loader: DataLoader,
                          ref_loader: DataLoader,
                          device: torch.device,
                          k: Tuple[int, int, int] = (1, 5, 10)
                          ) -> Dict[str, float]:

    # Calculate all embeddings of training set and test set
    embeddings_test, labels_test = get_embeddings_from_dataloader(test_loader, model, device)
    embeddings_ref, labels_ref = get_embeddings_from_dataloader(ref_loader, model, device)

    # Expand dimension for batch calculating
    embeddings_test = embeddings_test.unsqueeze(dim=0)  # [M x K] -> [1 x M x embedding_size]
    embeddings_ref = embeddings_ref.unsqueeze(dim=0)  # [N x K] -> [1 x N x embedding_size]
    labels_test = labels_test.unsqueeze(dim=1)  # [M] -> [M x 1]

    # Pairwise distance of all embeddings between test set and reference set
    distances: torch.Tensor = torch.cdist(embeddings_test, embeddings_ref, p=2).squeeze()  # [M x N]

    # Calculate precision_at_k on test set with k=1, k=5 and k=10
    metrics: Dict[str, float] = {}
    for i in k:
        metrics[f"average_precision_at_{i}"] = calculate_precision_at_k(distances,
                                                                        labels_test,
                                                                        labels_ref,
                                                                        k=i
                                                                        )
    # Calculate mean average precision (MAP)
    mean_average_precision: float = sum(precision_at_k for precision_at_k in metrics.values()) / len(metrics)
    metrics["mean_average_precision"] = mean_average_precision

    # Calculate top-1 and top-5 and top-10 accuracy
    for i in k:
        metrics[f"top_{i}_accuracy"] = calculate_topk_accuracy(distances,
                                                               labels_test,
                                                               labels_ref,
                                                               top_k=i
                                                               )
    # Calculate NMI score
    n_classes: int = len(test_loader.dataset.classes)
    metrics["normalized_mutual_information"] = calculate_normalized_mutual_information(
        embeddings_test.squeeze(), labels_test.squeeze(), n_classes
    )

    return metrics


def calculate_precision_at_k(distances: torch.Tensor,
                             labels_test: torch.Tensor,
                             labels_ref: torch.Tensor,
                             k: int
                             ) -> float:

    _, indices = distances.topk(k=k, dim=1, largest=False)  # indices shape: [M x k]

    y_pred = []
    for i in range(k):
        indices_at_k: torch.Tensor = indices[:, i]  # [M]
        y_pred_at_k: torch.Tensor = labels_ref[indices_at_k].unsqueeze(dim=1)  # [M x 1]
        y_pred.append(y_pred_at_k)

    y_pred: torch.Tensor = torch.hstack(y_pred)  # [M x k]
    labels_test = torch.hstack((labels_test,) * k)  # [M x k]

    precision_at_k: float = ((y_pred == labels_test).sum(dim=1) / k).mean().item() * 100
    return precision_at_k


def calculate_topk_accuracy(distances: torch.Tensor,
                            labels_test: torch.Tensor,
                            labels_ref: torch.Tensor,
                            top_k: int
                            ) -> float:

    _, indices = distances.topk(k=top_k, dim=1, largest=False)  # indices shape: [M x k]

    y_pred = []
    for i in range(top_k):
        indices_at_k: torch.Tensor = indices[:, i]  # [M]
        y_pred_at_k: torch.Tensor = labels_ref[indices_at_k].unsqueeze(dim=1)  # [M x 1]
        y_pred.append(y_pred_at_k)

    y_pred: torch.Tensor = torch.hstack(y_pred)  # [M x k]
    labels_test = torch.hstack((labels_test,) * top_k)  # [M x k]

    n_predictions: int = y_pred.shape[0]
    n_true_predictions: int = ((y_pred == labels_test).sum(dim=1) > 0).sum().item()
    topk_accuracy: float = n_true_predictions / n_predictions * 100
    return topk_accuracy


def calculate_normalized_mutual_information(embeddings: torch.Tensor,
                                            labels_test: torch.Tensor,
                                            n_classes: int
                                            ) -> float:
    embeddings = embeddings.cpu().numpy()
    y_test: np.ndarray = labels_test.cpu().numpy().astype(np.int)

    y_pred: np.ndarray = KMeans(n_clusters=n_classes).fit(embeddings).labels_
    NMI_score: float = normalized_mutual_info_score(y_test, y_pred)

    return NMI_score
