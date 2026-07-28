import os
import random

import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset


class MatDataset(Dataset):
    """Load one pansharpening dataset from MS_32/PAN_128/GT_128 folders."""

    def __init__(self, image_dir):
        self.image_dir = image_dir
        ms_dir = os.path.join(image_dir, "MS_32")
        self.image_names = sorted(
            name for name in os.listdir(ms_dir) if name.endswith(".mat")
        )

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        name = self.image_names[index]
        ms = scipy.io.loadmat(
            os.path.join(self.image_dir, "MS_32", name)
        )["ms0"]
        pan = scipy.io.loadmat(
            os.path.join(self.image_dir, "PAN_128", name)
        )["pan0"]
        gt = scipy.io.loadmat(
            os.path.join(self.image_dir, "GT_128", name)
        )["gt0"]
        return (
            np.asarray(ms, dtype=np.float32),
            np.asarray(pan, dtype=np.float32),
            np.asarray(gt, dtype=np.float32),
        )


class CombineMatDataset(Dataset):
    """Sample the four sensor datasets together with their one-hot task IDs."""

    def __init__(self, datasets, dataset_labels):
        if len(datasets) != len(dataset_labels):
            raise ValueError("datasets and dataset_labels must have equal length")
        self.datasets = datasets
        self.dataset_labels = dataset_labels
        self.length = min(len(dataset) for dataset in datasets)
        self.dataset_indices = [
            list(range(len(dataset))) for dataset in datasets
        ]

    def __len__(self):
        return self.length

    def shuffle(self):
        for indices in self.dataset_indices:
            random.shuffle(indices)

    def __getitem__(self, index):
        samples = []
        for dataset, indices, label in zip(
                self.datasets, self.dataset_indices, self.dataset_labels):
            one_hot = torch.zeros(len(self.dataset_labels))
            one_hot[label] = 1
            samples.append((dataset[indices[index]], one_hot))
        return tuple(samples)
