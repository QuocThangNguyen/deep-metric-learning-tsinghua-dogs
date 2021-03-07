import numpy as np
import torch
from torch.utils.data.sampler import Sampler
from collections import defaultdict

import random
from typing import List, DefaultDict, Dict, Iterator, Any


def _create_groups(targets: List[int], samples_per_class: int) -> DefaultDict[int, List[int]]:
    """
    Group samples' indexes according to their classes

    Args:
        targets: where i_th index stores i_th sample's group id

        samples_per_class: number of samples in each class

    Return:
        {
            class_0: [index_0, index_2, index_3, ...],
            class_1: [index_1, index_7, index_9, index_10...]
        }
    """
    class_to_idxs: DefaultDict[int, List[int]] = defaultdict(list)
    for idx, class_idx in enumerate(targets):
        class_to_idxs[class_idx].append(idx)

    # Get classes that have number of samples less than k
    classes_to_extend: List[int] = []
    for class_ in class_to_idxs:
        if len(class_to_idxs[class_]) < samples_per_class:
            classes_to_extend.append(class_)
            continue

    # For class that have less than k sample, we will extend that class
    # by duplicating random samples in class until the class has k samples
    for class_ in classes_to_extend:
        n_samples: int = len(class_to_idxs[class_])
        n_samples_to_extend = samples_per_class - n_samples

        random_samples: List[int] = []
        for _ in range(n_samples_to_extend):
            # Choose a random sample in class to duplicate
            random_sample: int = np.random.choice(class_to_idxs[class_])
            random_samples.append(random_sample)

        # Extend current class to have k samples
        class_to_idxs[class_].extend(random_samples)

    return class_to_idxs


class PKSampler(Sampler):
    """
    Randomly samples from a dataset while ensuring that each batch of size (p * k)
    includes samples from exactly p classes, with k samples for each class

    Parameters:
    -----------
    targets (list[int]): List where the i_th entry is the group_id/label of the i_th sample in the dataset.

    classes_per_batch (int): Number of classes to be sampled in a batch.

    samples_per_class (int): Number of samples per class in a batch.

    sequential_sampling (bool):
        If False: sample triplets in each batch, else sample images sequentially. Default False.
    """

    def __init__(self,
                 labels: List[int],
                 classes_per_batch: int,
                 samples_per_class: int,
                 sequential_sampling: bool = False
                 ):

        self.labels: List[int] = labels
        self.classes_per_batch: int = classes_per_batch
        self.samples_per_class: int = samples_per_class
        self.class_to_idxs: DefaultDict[int, List[int]] = _create_groups(labels, self.samples_per_class)

        self.sequential_sampling: bool = sequential_sampling

        # Ensures there are enough classes to sample from
        if len(self.class_to_idxs) < classes_per_batch:
            raise Exception(f"There are not enough classes to sample."
                            f"Got: class_to_idxs={self.class_to_idxs}, "
                            f"classes_per_batch={classes_per_batch}"
                            )

    def __iter__(self) -> Iterator[Any]:
        """
        Return index of images and targets to be sampled
        """
        if not self.sequential_sampling:
            # Shuffle sample within classes
            for key in self.class_to_idxs:
                random.shuffle(self.class_to_idxs[key])

            # Keep trach of the number of samples left for each classes
            class_to_n_samples: Dict[int, int] = {}
            for class_, sample_idxs in self.class_to_idxs.items():
                class_to_n_samples[class_] = len(sample_idxs)

            while len(class_to_n_samples) >= self.classes_per_batch:
                # Select p classes at random from valid remaining classes
                classes: List[int] = list(class_to_n_samples.keys())
                selected_class_idxs: List[int] = torch.multinomial(
                    torch.ones(len(classes)),
                    self.classes_per_batch
                ).tolist()

                for i in selected_class_idxs:
                    class_: int = classes[i]
                    # List of indexes of samples in a particular class
                    sample_idxs: List[int] = self.class_to_idxs[class_]
                    for _ in range(self.samples_per_class):
                        # Sequentially return an index of a sample in a class
                        sample_idx: int = len(sample_idxs) - class_to_n_samples[class_]
                        yield sample_idxs[sample_idx]
                        class_to_n_samples[class_] -= 1

                    # Don't sample from class if it has less than k samples remaning
                    if class_to_n_samples[class_] < self.samples_per_class:
                        class_to_n_samples.pop(class_)

        else:
            for i in range(len(self.labels)):
                yield i
