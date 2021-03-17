import torch
import torch.nn as nn
from typing import Tuple


# Modify from https://github.com/pytorch/vision/blob/master/references/similarity/loss.py


class TripletMarginLoss(nn.Module):
    def __init__(self, margin=1.0, p=2.0, sampling_type="batch_hard_triplets"):
        super().__init__()
        self.margin: float = margin
        self.p: float = p
        self.sampling_type: str = sampling_type

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, float]:
        if self.sampling_type == "batch_hard_triplets":
            return _batch_hard_triplets_loss(labels, embeddings, self.margin, self.p)
        elif self.sampling_type == "batch_hardest_triplets":
            return _batch_hardest_triplets_loss(labels, embeddings, self.margin, self.p)
        else:
            raise NotImplementedError(self.sampling_type)


def _batch_hard_triplets_loss(labels: torch.Tensor,
                              embeddings: torch.Tensor,
                              margin: float,
                              p: float
                              ) -> Tuple[torch.Tensor, float]:

    pairwise_distance: torch.Tensor = torch.cdist(embeddings, embeddings, p=p)

    anchor_positive_distance: torch.Tensor = pairwise_distance.unsqueeze(2)
    anchor_negative_distance: torch.Tensor = pairwise_distance.unsqueeze(1)

    # Indexes of all triplets
    mask: torch.Tensor = _get_triplet_masks(labels)
    # Calucalate triplet loss
    triplet_loss: torch.Tensor = mask.float() * (anchor_positive_distance - anchor_negative_distance + margin)

    # Remove negative loss (easy triplets)
    triplet_loss[triplet_loss < 0] = 0

    # Count number of positive triplets (where triplet_loss > 0)
    hard_triplets: torch.Tensor = triplet_loss[triplet_loss > 1e-16]
    n_hard_triplets: int = hard_triplets.size(0)

    # Total triplets (including positive and negative triplet)
    n_triplets: int = mask.sum().item()
    # Fraction of postive triplets in total
    fraction_hard_triplets: float = n_hard_triplets / (n_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = triplet_loss.sum() / (n_hard_triplets + 1e-16)
    return triplet_loss, fraction_hard_triplets


def _batch_hardest_triplets_loss(labels: torch.Tensor,
                                 embeddings: torch.Tensor,
                                 margin: float,
                                 p: float
                                 ) -> Tuple[torch.Tensor, int]:

    pairwise_distance: torch.Tensor = torch.cdist(embeddings, embeddings, p=p)
    # Indexes of all triplets
    mask_anchor_positive: torch.Tensor = _get_anchor_positive_mask(labels).float()
    # Distance between anchors and positives
    anchor_positive_distance: torch.Tensor = mask_anchor_positive * pairwise_distance

    # Hardest postive for every anchor
    hardest_positive_distance, _ = anchor_positive_distance.max(1, keepdim=True)

    mask_anchor_negative: torch.Tensor = _get_anchor_negative_mask(labels).float()
    # Add max value in each row to invalid negatives
    max_anchor_negative_distance, _ = pairwise_distance.max(dim=1, keepdim=True)
    anchor_negative_distance = pairwise_distance + max_anchor_negative_distance * (1.0 - mask_anchor_negative)

    # Hardest negative for every anchor
    hardest_negative_distance, _ = anchor_negative_distance.min(dim=1, keepdim=True)

    triplet_loss: torch.Tensor = hardest_positive_distance - hardest_negative_distance + margin
    triplet_loss[triplet_loss < 0] = 0
    triplet_loss = triplet_loss.mean()
    return triplet_loss, 0


def _get_triplet_masks(labels: torch.Tensor) -> torch.Tensor:
    # indices_equal is a square matrix, 1 in the diagonal and 0 everywhere else
    indices_equal: torch.Tensor = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    # indices_not_equal is inversed of indices_equal, 0 in the diagonal and 1 everywhere else
    indices_not_equal: torch.Tensor = ~indices_equal

    # convention: i (anchor index), j (positive index), k (negative index)
    i_not_equal_j: torch.Tensor = indices_not_equal.unsqueeze(2)
    i_not_equal_k: torch.Tensor = indices_not_equal.unsqueeze(1)
    j_not_equal_k: torch.Tensor = indices_not_equal.unsqueeze(0)
    # Check that anchor, positive, negative are distince to each other
    distinct_indices: torch.Tensor = (i_not_equal_j & i_not_equal_k) & j_not_equal_k

    label_equal: torch.Tensor = labels.unsqueeze(0) == labels.unsqueeze(1)
    i_equal_j: torch.Tensor = label_equal.unsqueeze(2)
    i_equal_k: torch.Tensor = label_equal.unsqueeze(1)

    # i_equal_j: indices of anchor-positive pairs
    # ~i_equal_k: indices of anchor-negative pairs
    # indices of valid triplets
    indices_triplets: torch.Tensor = i_equal_j & (~i_equal_k)
    # Make sure that anchor, positive, negative are distince to each other
    indices_triplets = indices_triplets & distinct_indices
    return indices_triplets


def _get_anchor_positive_mask(labels: torch.Tensor) -> torch.Tensor:
    # Check that i and j are distince
    indices_equal: torch.Tensor = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    indices_not_equal: torch.Tensor = ~indices_equal

    # Check anchor and negative
    labels_equal: torch.Tensor = labels.unsqueeze(0) == labels.unsqueeze(1)
    return labels_equal & indices_not_equal


def _get_anchor_negative_mask(labels: torch.Tensor) -> torch.BoolTensor:
    return labels.unsqueeze(0) != labels.unsqueeze(1)
