import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# Modify from https://github.com/idstcv/SoftTriple/blob/master/loss/SoftTriple.py


class SoftTripleLoss(nn.Module):
    def __init__(self,
                 n_classes: int,
                 embedding_size: int,
                 n_centers_per_class: int,
                 lambda_: float,
                 gamma: float,
                 tau: float,
                 margin: float,
                 device: torch.device
                 ):
        super().__init__()

        self.device: torch.device = device
        self.n_classes: int = n_classes
        self.embedding_size: int = embedding_size
        self.n_centers_per_class: int = n_centers_per_class

        self.lambda_: float = lambda_
        self.gamma: float = gamma
        self.tau: float = tau
        self.margin: float = margin

        # Each class has n centers
        self.centers: torch.Tensor = nn.Parameter(
            torch.rand(embedding_size, n_centers_per_class * n_classes)
        ).to(device)
        nn.init.kaiming_uniform_(self.centers, a=5**0.5)

        # weight for regularization term
        self.weight: torch.Tensor = torch.zeros(
            n_classes * n_centers_per_class,
            n_classes * n_centers_per_class,
            dtype=torch.long
        ).to(device)

        for i in range(n_classes):
            for j in range(n_centers_per_class):
                self.weight[
                    i * n_centers_per_class + j,
                    i * n_centers_per_class + j + 1:(i + 1) * n_centers_per_class
                ] = 1

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, int]:
        # labels shape: [n_classes]
        labels = labels.to(self.device)
        # centers shape: [embedding_size * (n_classes * n_centers_per_class)]
        centers = F.normalize(self.centers, p=2, dim=0)

        # Distance from each embedding to all centers
        # distances shape: [batch_size * n_classes * n_centers_per_class]
        distances = embeddings.matmul(centers).reshape(-1, self.n_classes, self.n_centers_per_class)

        # probabilities shape: [batch_size * n_classes * n_centers_per_class]
        probabilities = F.softmax(distances * self.gamma, dim=2)

        # Distance from each embedding to its TRUE center in a particular class
        # will depend on the distances betweeen it and all centers in that class.
        # relaxed_distances shape: [batch_size * n_classes]
        relaxed_distances = torch.sum(probabilities * distances, dim=2)

        margin = torch.zeros_like(relaxed_distances)
        margin[torch.arange(0, len(embeddings)), labels] = self.margin
        soft_triple_loss = F.cross_entropy(self.lambda_ * (relaxed_distances - margin), labels)

        if self.tau > 0 and self.n_centers_per_class > 1:
            # Distances between all pairs of centers
            distances_centers = centers.t().matmul(centers)
            dominator = torch.sum(torch.sqrt(2.0 + 1e-5 - 2. * distances_centers[self.weight]))
            denominator = (self.n_classes * self.n_centers_per_class * (self.n_centers_per_class - 1.))
            regularization = dominator / denominator
            return soft_triple_loss + self.tau * regularization, 0

        else:
            return soft_triple_loss, 0
