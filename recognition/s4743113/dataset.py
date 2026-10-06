"""ADNI data loading utilities.

The expected directory layout is::

    data/
        train/                 # ``training`` is also accepted
            <class_name>/
                <sample_id>_<slice>.jpg
                ...
        test/                  # ``testing`` is also accepted
            <class_name>/
                <sample_id>_<slice>.jpg
                ...

Files may be nested more deeply below a class directory.  The class name is
always taken from the first directory below ``train`` or ``test``.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split


_SLICE_NAME = re.compile(r"^(?P<sample>.+)_(?P<slice>\d+)$")
_IMAGE_SUFFIXES = {".jpg", ".jpeg"}


@dataclass(frozen=True)
class ADNISample:
    """The files and label belonging to one ADNI sample."""

    sample_id: str
    label: int
    slice_paths: tuple[Path, ...]
    slice_indices: tuple[int, ...]


def set_seed(seed: int) -> None:
    """Seed Python and PyTorch random-number generators."""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    """Give each data-loading worker a deterministic Python RNG state."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _find_split_directory(data_dir: Path, names: Sequence[str]) -> Path:
    wanted = {name.casefold() for name in names}
    matches = sorted(
        path
        for path in data_dir.iterdir()
        if path.is_dir() and path.name.casefold() in wanted
    )
    if not matches:
        choices = ", ".join(sorted(names))
        raise FileNotFoundError(
            f"Could not find a split directory ({choices}) inside {data_dir}"
        )
    return matches[0]


def _image_to_tensor(
    path: Path,
    image_size: tuple[int, int] | None,
) -> Tensor:
    """Read an image as an RGB float tensor in ``[0, 1]``."""

    with Image.open(path) as image:
        image = image.convert("RGB")
        if image_size is not None:
            # Public APIs conventionally specify (height, width), whereas PIL
            # expects (width, height).
            height, width = image_size
            image = image.resize((width, height), Image.Resampling.BILINEAR)
        width, height = image.size
        pixels = torch.frombuffer(
            bytearray(image.tobytes()), dtype=torch.uint8
        ).view(height, width, 3)

    return pixels.permute(2, 0, 1).contiguous().float().div_(255.0)


class ADNIDataset(Dataset[dict[str, object]]):
    """A dataset where all JPEG slices for a subject are one item.

    Each item is a dictionary containing:

    - ``images``: ``[slices, 3, height, width]`` float tensor
    - ``label``: integer class index
    - ``sample_id``: filename prefix preceding the final ``_<slice>``
    - ``slice_indices``: the numeric suffixes, in ascending order

    ``transform`` receives the complete stacked slice tensor, so random spatial
    transforms can be applied consistently to the sample rather than choosing
    a different transform for every slice.
    """

    def __init__(
        self,
        split_dir: str | Path,
        *,
        class_to_idx: Mapping[str, int] | None = None,
        image_size: tuple[int, int] | None = (224, 224),
        transform: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        self.split_dir = Path(split_dir)
        self.image_size = image_size
        self.transform = transform

        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"Dataset split does not exist: {self.split_dir}")

        image_paths = sorted(
            path
            for path in self.split_dir.rglob("*")
            if path.is_file() and path.suffix.casefold() in _IMAGE_SUFFIXES
        )
        if not image_paths:
            raise FileNotFoundError(f"No JPEG images found inside {self.split_dir}")

        class_names: set[str] = set()
        parsed: list[tuple[Path, str, str, int]] = []
        for path in image_paths:
            relative = path.relative_to(self.split_dir)
            if len(relative.parts) < 2:
                raise ValueError(
                    f"Image {path} is not inside a class directory. Expected "
                    f"{self.split_dir}/<class>/<sample>_<slice>.jpg"
                )

            match = _SLICE_NAME.fullmatch(path.stem)
            if match is None:
                raise ValueError(
                    f"Invalid image name {path.name!r}; expected "
                    "<sample>_<numeric_slice>.jpg"
                )

            class_name = relative.parts[0]
            sample_id = match.group("sample")
            slice_index = int(match.group("slice"))
            class_names.add(class_name)
            parsed.append((path, class_name, sample_id, slice_index))

        if class_to_idx is None:
            self.class_to_idx = {
                name: index for index, name in enumerate(sorted(class_names))
            }
        else:
            self.class_to_idx = dict(class_to_idx)
            unknown = class_names.difference(self.class_to_idx)
            if unknown:
                raise ValueError(
                    "Classes in the split are absent from class_to_idx: "
                    + ", ".join(sorted(unknown))
                )

        grouped: dict[tuple[str, str], list[tuple[int, Path]]] = defaultdict(list)
        for path, class_name, sample_id, slice_index in parsed:
            grouped[(class_name, sample_id)].append((slice_index, path))

        self.samples: list[ADNISample] = []
        for (class_name, sample_id), slices in sorted(grouped.items()):
            slices.sort(key=lambda item: (item[0], str(item[1])))
            indices = tuple(index for index, _ in slices)
            if len(indices) != len(set(indices)):
                raise ValueError(
                    f"Sample {sample_id!r} in class {class_name!r} has duplicate "
                    "slice indices"
                )
            self.samples.append(
                ADNISample(
                    sample_id=sample_id,
                    label=self.class_to_idx[class_name],
                    slice_paths=tuple(path for _, path in slices),
                    slice_indices=indices,
                )
            )

    @property
    def classes(self) -> list[str]:
        """Class names ordered by their integer label."""

        ordered_classes = sorted(
            self.class_to_idx.items(), key=lambda item: item[1]
        )
        return [name for name, _ in ordered_classes]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        images = torch.stack(
            [_image_to_tensor(path, self.image_size) for path in sample.slice_paths]
        )
        if self.transform is not None:
            images = self.transform(images)

        return {
            "images": images,
            "label": sample.label,
            "sample_id": sample.sample_id,
            "slice_indices": sample.slice_indices,
        }


def collate_adni_samples(
    items: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Pad samples to the largest slice count in a mini-batch.

    ``slice_mask[b, s]`` is true when ``images[b, s]`` is a real slice and
    false for padding.  A model can use this mask when pooling over slices.
    """

    if not items:
        raise ValueError("Cannot collate an empty batch")

    image_tensors = [item["images"] for item in items]
    if not all(
        isinstance(images, Tensor) and images.ndim == 4
        for images in image_tensors
    ):
        raise TypeError("Every item must contain a 4D 'images' tensor")

    images = [tensor for tensor in image_tensors if isinstance(tensor, Tensor)]
    max_slices = max(tensor.shape[0] for tensor in images)
    spatial_shapes = {tuple(tensor.shape[1:]) for tensor in images}
    if len(spatial_shapes) != 1:
        raise ValueError(
            "All images in a batch must have the same channel and spatial dimensions. "
            "Pass a fixed image_size or a resizing transform."
        )

    batch_shape = (len(images), max_slices, *images[0].shape[1:])
    batch = images[0].new_zeros(batch_shape)
    slice_mask = torch.zeros((len(images), max_slices), dtype=torch.bool)
    for batch_index, tensor in enumerate(images):
        slice_count = tensor.shape[0]
        batch[batch_index, :slice_count] = tensor
        slice_mask[batch_index, :slice_count] = True

    return {
        "images": batch,
        "labels": torch.tensor(
            [int(item["label"]) for item in items], dtype=torch.long
        ),
        "slice_mask": slice_mask,
        "sample_ids": [str(item["sample_id"]) for item in items],
        "slice_indices": [item["slice_indices"] for item in items],
    }


def create_adni_dataloaders(
    data_dir: str | Path | None = None,
    *,
    batch_size: int = 8,
    validation_fraction: float = 0.2,
    seed: int = 42,
    image_size: tuple[int, int] | None = (224, 224),
    train_transform: Callable[[Tensor], Tensor] | None = None,
    evaluation_transform: Callable[[Tensor], Tensor] | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create deterministic 80/20 train/validation and test loaders.

    The split happens after slices have been grouped, so every slice belonging
    to a sample is guaranteed to remain in the same subset.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    set_seed(seed)
    # Anchor the default to this source file, not to whichever directory was
    # active when Python/Jupyter was launched.  This remains portable if the
    # whole s4743113 directory is moved.
    data_path = (
        Path(__file__).resolve().parent / "data"
        if data_dir is None
        else Path(data_dir)
    )
    if not data_path.is_dir():
        raise FileNotFoundError(f"ADNI data directory does not exist: {data_path}")

    train_dir = _find_split_directory(data_path, ("train", "training"))
    test_dir = _find_split_directory(data_path, ("test", "testing"))

    # Validation must not receive random training augmentation.  The two
    # datasets have identical ordering, so the same generated indices can be
    # applied to each view.
    training_dataset = ADNIDataset(
        train_dir, image_size=image_size, transform=train_transform
    )
    validation_dataset = ADNIDataset(
        train_dir,
        class_to_idx=training_dataset.class_to_idx,
        image_size=image_size,
        transform=evaluation_transform,
    )
    test_dataset = ADNIDataset(
        test_dir,
        class_to_idx=training_dataset.class_to_idx,
        image_size=image_size,
        transform=evaluation_transform,
    )

    sample_count = len(training_dataset)
    if sample_count < 2:
        raise ValueError("At least two grouped training samples are required")
    validation_count = max(1, round(sample_count * validation_fraction))
    training_count = sample_count - validation_count
    if training_count == 0:
        raise ValueError("validation_fraction leaves no samples for training")

    split_generator = torch.Generator().manual_seed(seed)
    training_subset, validation_index_subset = random_split(
        training_dataset,
        [training_count, validation_count],
        generator=split_generator,
    )

    # Reuse the exact validation indices with the non-augmented dataset view.
    validation_indices = validation_index_subset.indices
    validation_subset = torch.utils.data.Subset(validation_dataset, validation_indices)

    common_loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
        "worker_init_fn": _seed_worker,
        "collate_fn": collate_adni_samples,
    }
    train_loader = DataLoader(
        training_subset,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 1),
        **common_loader_options,
    )
    validation_loader = DataLoader(
        validation_subset,
        shuffle=False,
        **common_loader_options,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **common_loader_options,
    )

    return train_loader, validation_loader, test_loader
