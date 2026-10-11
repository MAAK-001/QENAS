"""PyTorch datasets/loaders over preloaded arrays, and the split bundle used by all stages.

Test isolation: :class:`DataBundle` loads only the train and validation splits.
The test split is materialised exclusively by :meth:`DataBundle.load_test`, which
the pipeline calls once, after the final model is trained and every choice is fixed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from ..utils.seed import derive_seed
from .augmentation import augment_pair
from .discovery import Sample
from .preprocessing import channel_stats, preload_split, trivial_foreground_dice


class ArraySegDataset(Dataset):
    def __init__(self, images: np.ndarray, masks: np.ndarray, mean: Sequence[float], std: Sequence[float],
                 augment: Optional[Dict[str, Any]] = None, seed: int = 0):
        self.images, self.masks = images, masks
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.augment = augment
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int):
        # The training sampler encodes the epoch into the index (epoch * N + i) so that the
        # augmentation stream is correct even inside persistent data-loader worker processes.
        epoch, i = divmod(int(idx), len(self))
        img = torch.from_numpy(np.ascontiguousarray(self.images[i])).permute(2, 0, 1).float().div_(255.0)
        mask = torch.from_numpy(np.ascontiguousarray(self.masks[i])).unsqueeze(0).float()
        if self.augment is not None:
            rng = np.random.default_rng(derive_seed(self.seed, "aug", epoch, i))
            img, mask = augment_pair(img, mask, self.augment, rng)
        img = (img - self.mean) / self.std
        return img, mask


class EpochShuffleSampler(Sampler[int]):
    """Deterministic per-epoch permutation (resumable: depends only on seed and epoch)."""

    def __init__(self, n: int, seed: int):
        self.n, self.seed, self.epoch = int(n), int(seed), 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(derive_seed(self.seed, "shuffle", self.epoch))
        offset = self.epoch * self.n
        return iter((offset + rng.permutation(self.n)).tolist())

    def __len__(self) -> int:
        return self.n


class DataBundle:
    """Train/val arrays + training-split normalisation statistics; test is loaded lazily and only once."""

    def __init__(self, splits: Dict[str, List[Sample]], cfg: Dict[str, Any]):
        self.cfg = cfg
        self.splits = splits
        d = cfg["data"]
        self.size_hw = tuple(d["image_size"])
        self.threshold = int(d["mask_threshold"])
        self.dataset = d["dataset"]
        self.cache_dir = Path(cfg["paths"]["cache"]) / "preprocessed"
        self.train_images, self.train_masks = self._load("train")
        self.val_images, self.val_masks = self._load("val")
        self.mean, self.std = channel_stats(self.train_images)
        self.val_trivial_dice = trivial_foreground_dice(self.val_masks)

    def _load(self, split: str):
        return preload_split(self.splits[split], self.size_hw, self.threshold, self.dataset, split, self.cache_dir)

    def _loader(self, ds: ArraySegDataset, batch_size: int, train: bool, seed: int) -> DataLoader:
        nw = int(self.cfg["data"]["num_workers"])
        if train:
            sampler = EpochShuffleSampler(len(ds), seed)
            drop_last = len(ds) % batch_size == 1   # avoid a single-sample BatchNorm batch
            return DataLoader(ds, batch_size=batch_size, sampler=sampler, drop_last=drop_last,
                              num_workers=nw, pin_memory=torch.cuda.is_available(),
                              persistent_workers=nw > 0)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=nw,
                          pin_memory=torch.cuda.is_available(), persistent_workers=nw > 0)

    def train_loader(self, batch_size: int, seed: int) -> DataLoader:
        ds = ArraySegDataset(self.train_images, self.train_masks, self.mean, self.std,
                             augment=self.cfg["augmentation"], seed=seed)
        return self._loader(ds, batch_size, train=True, seed=seed)

    def train_eval_loader(self, batch_size: int) -> DataLoader:
        """Training images without augmentation (for reporting final train metrics)."""
        ds = ArraySegDataset(self.train_images, self.train_masks, self.mean, self.std)
        return self._loader(ds, batch_size, train=False, seed=0)

    def val_loader(self, batch_size: int) -> DataLoader:
        ds = ArraySegDataset(self.val_images, self.val_masks, self.mean, self.std)
        return self._loader(ds, batch_size, train=False, seed=0)

    def zico_batches(self, batch_size: int, n_batches: int, seed: int):
        """The first ``n_batches`` training batches of epoch 0 (shuffled and augmented exactly as in training).
        Fixed for the whole search, as the official ZiCo search fixes ``trainbatches`` once."""
        loader = self.train_loader(batch_size, seed)
        set_loader_epoch(loader, 0)
        out = []
        for x, y in loader:
            out.append((x, y))
            if len(out) == n_batches:
                break
        if len(out) < n_batches:
            raise RuntimeError(f"ZiCo needs {n_batches} training batches of size {batch_size}; "
                               f"the training split has only {len(self.train_images)} images")
        return out

    def load_test(self, batch_size: int) -> DataLoader:
        """Materialise the held-out test split. Must only be called by the final test stage."""
        imgs, masks = self._load("test")
        ds = ArraySegDataset(imgs, masks, self.mean, self.std)
        return self._loader(ds, batch_size, train=False, seed=0)


def set_loader_epoch(loader: DataLoader, epoch: int) -> None:
    if isinstance(loader.sampler, EpochShuffleSampler):
        loader.sampler.set_epoch(epoch)
