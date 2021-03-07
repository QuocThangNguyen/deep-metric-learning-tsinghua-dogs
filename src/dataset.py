import numpy as np
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
from PIL import ImageFile

from itertools import groupby
from typing import List, Tuple, Dict


ImageFile.LOAD_TRUNCATED_IMAGES = True


class Dataset(ImageFolder):
    def __init__(self, images_dir: str, transform=None):
        super().__init__(images_dir, transform=transform)

        self.idx_to_class: Dict[int, str] = {
            idx: class_name
            for class_name, idx in self.class_to_idx.items()
        }

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target


def get_subset_from_dataset(dataset: Dataset, n_samples_per_class: int) -> Dataset:
    # samples + corresponding indices in database_set
    samples: List[Tuple[int, str, int]] = [(i, *sample) for i, sample in enumerate(dataset.samples)]
    group_by_class_idx = groupby(samples, key=lambda sample: sample[2])  # group by class_index

    indices: List[int] = []
    for _, group in group_by_class_idx:
        group: List = list(group)
        indices_in_same_class, _, _ = zip(*group)
        indices_in_same_class: List[int] = np.random.choice(
            indices_in_same_class, size=n_samples_per_class, replace=False
        ).tolist()
        indices.extend(indices_in_same_class)

    subset: Dataset = Subset(dataset, indices=indices)
    return subset
