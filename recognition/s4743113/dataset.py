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
from torch.utils.data import DataLoader, Dataset


_SLICE_NAME = re.compile(r"^(?P<sample>.+)_(?P<slice>\d+)$")
_IMAGE_SUFFIXES = {".jpg", ".jpeg"}


@dataclass(frozen=True)
class ADNISample:
    """The files and label belonging to one ADNI sample."""

    sample_id: str
    label: int
    slice_paths: tuple[Path, ...]
    slice_indices: tuple[int, ...]


@dataclass(frozen=True)
class ADNISlice:
    """One labelled MRI slice belonging to an ADNI sample."""

    sample_id: str
    label: int
    path: Path
    slice_index: int


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


class ADNISubjectIndex:
    """Index slice paths by subject before creating slice datasets."""

    def __init__(
        self,
        split_dir: str | Path,
        *,
        class_to_idx: Mapping[str, int] | None = None,
        image_size: tuple[int, int] | None = (224, 224),
    ) -> None:
        self.split_dir = Path(split_dir)
        self.image_size = image_size

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


class ADNISliceDataset(Dataset[dict[str, object]]):
    """Expose slices from selected subjects as independent examples.

    Subject indices are chosen before this dataset is created. Therefore,
    flattening the selected subjects into slices cannot place neighbouring
    slices from one subject in both the training and validation subsets.
    """

    def __init__(
        self,
        subject_dataset: ADNISubjectIndex,
        subject_indices: Sequence[int] | None = None,
        *,
        transform: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        self.class_to_idx = dict(subject_dataset.class_to_idx)
        self.image_size = subject_dataset.image_size
        self.transform = transform

        if subject_indices is None:
            selected_indices = tuple(range(len(subject_dataset)))
        else:
            selected_indices = tuple(int(index) for index in subject_indices)

        if len(selected_indices) != len(set(selected_indices)):
            raise ValueError("subject_indices contains duplicate indices")
        if any(index < 0 or index >= len(subject_dataset) for index in selected_indices):
            raise IndexError("subject index is outside the dataset")

        self.subject_ids = tuple(
            subject_dataset.samples[index].sample_id for index in selected_indices
        )
        self.slices: list[ADNISlice] = []
        for subject_index in selected_indices:
            sample = subject_dataset.samples[subject_index]
            self.slices.extend(
                ADNISlice(
                    sample_id=sample.sample_id,
                    label=sample.label,
                    path=path,
                    slice_index=slice_index,
                )
                for path, slice_index in zip(
                    sample.slice_paths,
                    sample.slice_indices,
                )
            )

    @property
    def classes(self) -> list[str]:
        """Class names ordered by their integer label."""

        ordered_classes = sorted(
            self.class_to_idx.items(), key=lambda item: item[1]
        )
        return [name for name, _ in ordered_classes]

    @property
    def subject_count(self) -> int:
        return len(self.subject_ids)

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = self.slices[index]
        image = _image_to_tensor(item.path, self.image_size)
        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": item.label,
            "sample_id": item.sample_id,
            "slice_index": item.slice_index,
        }


def create_adni_slice_dataloaders(
    data_dir: str | Path | None = None,
    *,
    batch_size: int = 48,
    validation_fraction: float = 0.2,
    seed: int = 42,
    image_size: tuple[int, int] | None = (224, 224),
    train_transform: Callable[[Tensor], Tensor] | None = None,
    evaluation_transform: Callable[[Tensor], Tensor] | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create slice loaders after a deterministic subject-level split.

    Training subjects are split 80/20 before their records are flattened into
    individual slices. All slices from a subject therefore stay entirely in
    either training or validation, while ConvNeXt still receives one 2D slice
    and one label for each dataset item. The provided test split is flattened
    in the same way and remains separate from both subsets.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    set_seed(seed)
    data_path = (
        Path(__file__).resolve().parent / "data"
        if data_dir is None
        else Path(data_dir)
    )
    if not data_path.is_dir():
        raise FileNotFoundError(f"ADNI data directory does not exist: {data_path}")

    train_dir = _find_split_directory(data_path, ("train", "training"))
    test_dir = _find_split_directory(data_path, ("test", "testing"))
    training_subjects = ADNISubjectIndex(train_dir, image_size=image_size)
    test_subjects = ADNISubjectIndex(
        test_dir,
        class_to_idx=training_subjects.class_to_idx,
        image_size=image_size,
    )

    training_identities = {
        (sample.label, sample.sample_id) for sample in training_subjects.samples
    }
    test_identities = {
        (sample.label, sample.sample_id) for sample in test_subjects.samples
    }
    if training_identities & test_identities:
        raise RuntimeError("A subject appears in both the provided train and test splits")

    subject_count = len(training_subjects)
    if subject_count < 2:
        raise ValueError("At least two grouped training subjects are required")
    validation_count = max(1, round(subject_count * validation_fraction))
    training_count = subject_count - validation_count
    if training_count == 0:
        raise ValueError("validation_fraction leaves no subjects for training")

    permutation = torch.randperm(
        subject_count,
        generator=torch.Generator().manual_seed(seed),
    ).tolist()
    training_indices = permutation[:training_count]
    validation_indices = permutation[training_count:]

    training_slices = ADNISliceDataset(
        training_subjects,
        training_indices,
        transform=train_transform,
    )
    validation_slices = ADNISliceDataset(
        training_subjects,
        validation_indices,
        transform=evaluation_transform,
    )
    test_slices = ADNISliceDataset(
        test_subjects,
        transform=evaluation_transform,
    )

    if set(training_slices.subject_ids) & set(validation_slices.subject_ids):
        raise RuntimeError("A subject appears in both training and validation")

    common_loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
        "worker_init_fn": _seed_worker,
    }
    train_loader = DataLoader(
        training_slices,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 1),
        **common_loader_options,
    )
    validation_loader = DataLoader(
        validation_slices,
        shuffle=False,
        **common_loader_options,
    )
    test_loader = DataLoader(
        test_slices,
        shuffle=False,
        **common_loader_options,
    )

    return train_loader, validation_loader, test_loader
