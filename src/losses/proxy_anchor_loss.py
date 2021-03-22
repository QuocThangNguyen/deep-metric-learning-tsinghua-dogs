import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .proxy_nca_loss import binarize_and_smooth_labels


# Modify from: https://github.com/tjddus9597/Proxy-Anchor-CVPR2020/blob/master/code/losses.py


class ProxyAnchorLoss(nn.Module):
    def __init__(self, n_classes: int, embedding_size: int, margin: float, alpha: float, device: torch.device):
        super().__init__()

        self.device: torch.device = device
        self.n_classes: int = n_classes
        self.embedding_size: int = embedding_size

        # shape: [n_classes * embedding_size]
        self.proxies = nn.Parameter(torch.rand(n_classes, embedding_size)).to(device)
        nn.init.kaiming_normal_(self.proxies, mode="fan_out")

        self.margin: float = margin
        self.alpha: float = alpha

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, int]:
        # proxies shape: [n_classes * embedding_size]
        proxies: torch.Tensor = F.normalize(self.proxies, p=2, dim=1)

        # cosine_distances shape: [batch_size * n_classes]
        cosine_distances = F.linear(embeddings, proxies)
        # positive_exp shape: [batch_size * n_classes]
        positive_exp = torch.exp(-self.alpha * (cosine_distances - self.margin))
        # negative_exp shape: [batch_size * n_classes]
        negative_exp = torch.exp(self.alpha * (cosine_distances + self.margin))

        # shape: [batch_size * n_classes]
        labels_onehot: torch.Tensor = binarize_and_smooth_labels(
            labels, self.n_classes, smoothing_factor=0, device=self.device
        )
        # Indices of all positive proxies in batch, shape [n_classes]
        distinct_proxies_indices = torch.nonzero(labels_onehot.sum(dim=0) != 0).squeeze(dim=1)
        n_distinct_proxies: int = len(distinct_proxies_indices)

        # positive_distances: distance from a pariticular proxy to all positive samples in a batch
        # shape: [n_classes]
        sum_positive_distances = torch.where(
            labels_onehot == 1, positive_exp, torch.zeros_like(positive_exp)
        ).sum(dim=0)

        # negative distances: distance from a particular proxy to all negative samples in a batch
        # shape: [n_classes]
        sum_negative_distances = torch.where(
            labels_onehot == 0, negative_exp, torch.zeros_like(negative_exp)
        ).sum(dim=0)

        positive_term = torch.log(1 + sum_positive_distances).sum() / n_distinct_proxies
        negative_term = torch.log(1 + sum_negative_distances).sum() / self.n_classes
        proxy_anchor_loss: torch.Tensor = positive_term + negative_term
        return proxy_anchor_loss, 0
