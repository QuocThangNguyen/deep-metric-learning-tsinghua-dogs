import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
from typing import Tuple


# Modify from: https://github.com/dichotomies/proxy-nca/blob/master/proxynca.py


def binarize_and_smooth_labels(labels: torch.Tensor,
                               n_classes: int,
                               smoothing_factor: float,
                               device: torch.device
                               ) -> torch.FloatTensor:
    labels = labels.cpu().numpy()
    labels = label_binarize(labels, classes=range(0, n_classes))
    labels = labels * (1 - smoothing_factor)
    labels[labels == 0] = smoothing_factor / (n_classes - 1)
    labels = torch.from_numpy(labels).float().to(device)
    return labels


class ProxyNCALoss(nn.Module):
    def __init__(self,
                 n_classes: int,
                 embedding_size: int,
                 embedding_scale: float,
                 proxy_scale: float,
                 smoothing_factor: float,
                 device: torch.device,
                 ):
        super().__init__()

        self.device: torch.device = device
        self.n_classes: int = n_classes
        self.embedding_size: int = embedding_size
        self.embedding_scale: float = embedding_scale

        self.proxies = nn.Parameter(torch.randn(n_classes, embedding_size) / 8).to(device)
        self.proxy_scale: float = proxy_scale
        self.smoothing_factor: float = smoothing_factor

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, int]:
        # embeddings shape: [batch_size * embedding_size]
        embeddings = embeddings * self.embedding_scale
        # proxies shape: [n_classes * embedding_size]
        proxies: torch.Tensor = F.normalize(self.proxies, p=2, dim=1) * self.proxy_scale
        # distances shape: [batch_size * n_classes]
        distances: torch.Tensor = torch.cdist(embeddings, proxies).square()

        # labels shape: [batch_size * n_classes]
        labels = binarize_and_smooth_labels(labels, self.n_classes, self.smoothing_factor, self.device)
        proxy_nca_loss: torch.Tensor = (-labels * F.log_softmax(-distances, dim=1)).sum(dim=1)

        return proxy_nca_loss.mean(), 0
