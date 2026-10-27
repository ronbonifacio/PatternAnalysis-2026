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
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler


_SLICE_NAME = re.compile(r"^(?P<sample>.+)_(?P<slice>\d+)$")
_IMAGE_SUFFIXES = {".jpg", ".jpeg"}
_FOREGROUND_THRESHOLD = 10 / 255


@dataclass(frozen=True)
class ADNISample:
    """The files and label belonging to one ADNI MRI scan."""

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
    """Read, foreground-centre and resize an image to an RGB tensor."""

    with Image.open(path) as image:
        image = image.convert("RGB")
        if image_size is not None:
            height, width = image_size
            image = _foreground_centered_crop(image, (height, width))
            if image.size != (width, height):
                # Public APIs specify (height, width), whereas PIL expects
                # (width, height).
                image = image.resize((width, height), Image.Resampling.BILINEAR)
        width, height = image.size
        pixels = torch.frombuffer(
            bytearray(image.tobytes()), dtype=torch.uint8
        ).view(height, width, 3)

    return pixels.permute(2, 0, 1).contiguous().float().div_(255.0)


def _foreground_centered_crop(
    image: Image.Image,
    crop_size: tuple[int, int],
) -> Image.Image:
    """Take a constant-size crop centred on non-black anatomy.

    The crop size stays fixed rather than fitting each brain tightly, so
    anatomical scale differences are preserved. JPEG compression can make the
    nominally black background slightly non-zero, hence the small threshold.
    """

    target_height, target_width = crop_size
    crop_width = min(target_width, image.width)
    crop_height = min(target_height, image.height)

    grayscale = image.convert("L")
    threshold = round(_FOREGROUND_THRESHOLD * 255)
    foreground = grayscale.point(lambda value: 255 if value > threshold else 0)
    bounding_box = foreground.getbbox()

    if bounding_box is None:
        centre_x = image.width / 2
        centre_y = image.height / 2
    else:
        left, top, right, bottom = bounding_box
        centre_x = (left + right) / 2
        centre_y = (top + bottom) / 2

    crop_left = round(centre_x - crop_width / 2)
    crop_top = round(centre_y - crop_height / 2)
    crop_left = min(max(crop_left, 0), image.width - crop_width)
    crop_top = min(max(crop_top, 0), image.height - crop_height)

    return image.crop(
        (
            crop_left,
            crop_top,
            crop_left + crop_width,
            crop_top + crop_height,
        )
    )


def _calculate_foreground_statistics(
    samples: Sequence[ADNISample],
    subject_indices: Sequence[int],
    image_size: tuple[int, int] | None,
) -> tuple[float, float]:
    """Calculate foreground mean/std using training subjects only."""

    pixel_sum = 0.0
    squared_pixel_sum = 0.0
    pixel_count = 0

    for subject_index in subject_indices:
        for path in samples[subject_index].slice_paths:
            # ADNI JPEGs are grayscale replicated across RGB, so one channel
            # contains all unique intensity information.
            pixels = _image_to_tensor(path, image_size)[0]
            foreground = pixels[pixels > _FOREGROUND_THRESHOLD].double()
            pixel_sum += foreground.sum().item()
            squared_pixel_sum += foreground.square().sum().item()
            pixel_count += foreground.numel()

    if pixel_count == 0:
        raise RuntimeError("Training images contain no foreground pixels")

    mean = pixel_sum / pixel_count
    variance = max(squared_pixel_sum / pixel_count - mean**2, 0.0)
    standard_deviation = variance**0.5
    if standard_deviation == 0.0:
        raise RuntimeError("Training foreground has zero standard deviation")
    return mean, standard_deviation


class ADNISubjectIndex:
    """Index slice paths by scan before creating slice datasets."""

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

    Scan indices are chosen before this dataset is created. Therefore,
    flattening the selected scans into slices cannot place neighbouring slices
    from one scan in both the training and validation subsets.
    """

    def __init__(
        self,
        subject_dataset: ADNISubjectIndex,
        subject_indices: Sequence[int] | None = None,
        *,
        foreground_statistics: tuple[float, float] | None = None,
        transform: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        self.class_to_idx = dict(subject_dataset.class_to_idx)
        self.image_size = subject_dataset.image_size
        self.foreground_statistics = foreground_statistics
        self.transform = transform

        if foreground_statistics is not None:
            mean, standard_deviation = foreground_statistics
            if standard_deviation <= 0:
                raise ValueError("foreground standard deviation must be positive")
            self.foreground_mean = float(mean)
            self.foreground_standard_deviation = float(standard_deviation)
        else:
            self.foreground_mean = None
            self.foreground_standard_deviation = None

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
        if self.foreground_statistics is not None:
            foreground = image.amax(dim=0, keepdim=True) > _FOREGROUND_THRESHOLD
            image = torch.where(
                foreground,
                (image - self.foreground_mean)
                / self.foreground_standard_deviation,
                0.0,
            )

        return {
            "image": image,
            "label": item.label,
            "sample_id": item.sample_id,
            "slice_index": item.slice_index,
        }


class RandomSlicesPerScanSampler(Sampler[int]):
    """Select a new random subset of each scan's slices every epoch."""

    def __init__(
        self,
        dataset: ADNISliceDataset,
        slices_per_scan: int,
        *,
        seed: int,
    ) -> None:
        if slices_per_scan <= 0:
            raise ValueError("slices_per_scan must be positive")

        grouped_indices: dict[tuple[int, str], list[int]] = defaultdict(list)
        for index, item in enumerate(dataset.slices):
            grouped_indices[(item.label, item.sample_id)].append(index)

        if not grouped_indices:
            raise ValueError("Cannot sample slices from an empty dataset")

        self.scan_indices = tuple(
            tuple(indices) for indices in grouped_indices.values()
        )
        self.slices_per_scan = slices_per_scan
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1

        selected_indices: list[int] = []
        for scan_indices in self.scan_indices:
            count = min(self.slices_per_scan, len(scan_indices))
            selection = torch.randperm(
                len(scan_indices),
                generator=generator,
            )[:count]
            selected_indices.extend(scan_indices[index] for index in selection)

        order = torch.randperm(
            len(selected_indices),
            generator=generator,
        ).tolist()
        return iter(selected_indices[index] for index in order)

    def __len__(self) -> int:
        return sum(
            min(self.slices_per_scan, len(scan_indices))
            for scan_indices in self.scan_indices
        )


def create_adni_slice_dataloaders(
    data_dir: str | Path | None = None,
    *,
    batch_size: int = 48,
    validation_fraction: float = 0.2,
    seed: int = 42,
    training_seed: int | None = None,
    training_slices_per_scan: int | None = None,
    image_size: tuple[int, int] | None = (224, 224),
    train_transform: Callable[[Tensor], Tensor] | None = None,
    evaluation_transform: Callable[[Tensor], Tensor] | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create slice loaders after a deterministic scan-level split.

    Training scans are split 80/20 before their records are flattened into
    individual slices. All slices from a scan therefore stay entirely in
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
    if training_slices_per_scan is not None and training_slices_per_scan <= 0:
        raise ValueError("training_slices_per_scan must be positive")

    if training_seed is None:
        training_seed = seed

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
        raise RuntimeError(
            "A scan appears in both the provided train and test splits"
        )

    subject_count = len(training_subjects)
    if subject_count < 2:
        raise ValueError("At least two grouped training scans are required")
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
    foreground_statistics = _calculate_foreground_statistics(
        training_subjects.samples,
        training_indices,
        image_size,
    )

    training_slices = ADNISliceDataset(
        training_subjects,
        training_indices,
        foreground_statistics=foreground_statistics,
        transform=train_transform,
    )
    validation_slices = ADNISliceDataset(
        training_subjects,
        validation_indices,
        foreground_statistics=foreground_statistics,
        transform=evaluation_transform,
    )
    test_slices = ADNISliceDataset(
        test_subjects,
        foreground_statistics=foreground_statistics,
        transform=evaluation_transform,
    )

    if set(training_slices.subject_ids) & set(validation_slices.subject_ids):
        raise RuntimeError("A scan appears in both training and validation")

    common_loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        # Training, validation and testing use separate loaders. Do not keep
        # one loader's workers alive while the next loader starts another
        # worker pool, as this can exceed the Slurm job's host-memory limit.
        "persistent_workers": False,
        "worker_init_fn": _seed_worker,
    }
    training_sampler = None
    if training_slices_per_scan is not None:
        training_sampler = RandomSlicesPerScanSampler(
            training_slices,
            training_slices_per_scan,
            seed=training_seed,
        )

    train_loader = DataLoader(
        training_slices,
        shuffle=training_sampler is None,
        sampler=training_sampler,
        generator=torch.Generator().manual_seed(training_seed + 1),
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
