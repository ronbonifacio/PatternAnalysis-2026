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

import json
import math
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
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler


_SLICE_NAME = re.compile(r"^(?P<sample>.+)_(?P<slice>\d+)$")
_PATIENT_NAME = re.compile(r"ADNI_(?P<patient>\d{3}_S_\d{4})")
_IMAGE_SUFFIXES = {".jpg", ".jpeg"}
_FOREGROUND_THRESHOLD = 10 / 255
FULL_FOREGROUND_CROP_SIZE = (224, 224)


@dataclass(frozen=True)
class ADNISample:
    """The files and label belonging to one ADNI MRI scan."""

    sample_id: str
    label: int
    slice_paths: tuple[Path, ...]
    slice_indices: tuple[int, ...]
    patient_id: str


@dataclass(frozen=True)
class ADNISlice:
    """One labelled MRI slice and its within-scan context."""

    sample_id: str
    label: int
    path: Path
    context_paths: tuple[Path, Path, Path]
    slice_index: int
    patient_id: str


def set_seed(seed: int) -> None:
    """Seed Python and PyTorch random-number generators."""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resize_tensor_image(image: Tensor, *, size: int) -> Tensor:
    """Resize a CHW tensor to a square without taking another crop."""

    if image.shape[-2:] == (size, size):
        return image
    return F.interpolate(
        image.unsqueeze(0),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0)


class MRITrainAugmentation:
    """Apply shared spatial and intensity variation to a 2D/2.5D MRI input."""

    def __init__(
        self,
        *,
        size: int,
        rotation_degrees: float,
        translation_fraction: float,
        scale_jitter: float,
        intensity_jitter: float,
        noise_standard_deviation: float,
        horizontal_flip_probability: float = 0.0,
        gamma_jitter: float = 0.0,
    ) -> None:
        self.size = size
        self.rotation_degrees = rotation_degrees
        self.translation_fraction = translation_fraction
        self.scale_jitter = scale_jitter
        self.intensity_jitter = intensity_jitter
        self.noise_standard_deviation = noise_standard_deviation
        self.horizontal_flip_probability = horizontal_flip_probability
        self.gamma_jitter = gamma_jitter

    @staticmethod
    def _uniform(low: float, high: float) -> float:
        return torch.empty(()).uniform_(low, high).item()

    def __call__(self, image: Tensor) -> Tensor:
        image = resize_tensor_image(image, size=self.size)
        if torch.rand(()).item() < self.horizontal_flip_probability:
            image = image.flip(-1)

        angle = math.radians(
            self._uniform(-self.rotation_degrees, self.rotation_degrees)
        )
        desired_scale = self._uniform(
            1.0 - self.scale_jitter,
            1.0 + self.scale_jitter,
        )
        translation_x = 2.0 * self._uniform(
            -self.translation_fraction,
            self.translation_fraction,
        )
        translation_y = 2.0 * self._uniform(
            -self.translation_fraction,
            self.translation_fraction,
        )
        cosine = math.cos(angle) / desired_scale
        sine = math.sin(angle) / desired_scale
        theta = image.new_tensor(
            [
                [cosine, -sine, translation_x],
                [sine, cosine, translation_y],
            ]
        ).unsqueeze(0)
        grid = F.affine_grid(
            theta,
            size=(1, image.shape[0], self.size, self.size),
            align_corners=False,
        )
        image = F.grid_sample(
            image.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).squeeze(0)

        foreground = image.amax(dim=0, keepdim=True) > _FOREGROUND_THRESHOLD
        foreground_values = image.masked_select(foreground.expand_as(image))
        if foreground_values.numel():
            if self.gamma_jitter:
                gamma = self._uniform(
                    1.0 - self.gamma_jitter,
                    1.0 + self.gamma_jitter,
                )
                image = image.clamp_min(0.0).pow(gamma)
                foreground_values = image.masked_select(
                    foreground.expand_as(image)
                )
            foreground_mean = foreground_values.mean()
            contrast = self._uniform(
                1.0 - self.intensity_jitter,
                1.0 + self.intensity_jitter,
            )
            gain = self._uniform(
                1.0 - self.intensity_jitter,
                1.0 + self.intensity_jitter,
            )
            image = (image - foreground_mean) * contrast + foreground_mean
            image = image * gain
            if self.noise_standard_deviation:
                noise = torch.randn(
                    (1, self.size, self.size),
                    dtype=image.dtype,
                    device=image.device,
                ) * self.noise_standard_deviation
                image = image + noise
            image = torch.where(foreground, image.clamp(0.0, 1.0), 0.0)

        return image


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


def _load_patient_ids(metadata_path: Path) -> dict[str, str]:
    """Map each ADNI image/scan identifier to its participant identifier."""

    if not metadata_path.is_file():
        raise FileNotFoundError(f"ADNI metadata does not exist: {metadata_path}")
    with metadata_path.open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    if not isinstance(metadata, dict):
        raise ValueError("ADNI metadata must be a JSON object keyed by scan ID")

    patient_ids: dict[str, str] = {}
    for sample_id, record in metadata.items():
        if not isinstance(record, dict) or not isinstance(record.get("raw"), str):
            raise ValueError(f"Metadata entry {sample_id!r} has no raw image path")
        match = _PATIENT_NAME.search(record["raw"])
        if match is None:
            raise ValueError(
                f"Could not read an ADNI patient ID from metadata entry "
                f"{sample_id!r}"
            )
        patient_ids[str(sample_id)] = match.group("patient")
    return patient_ids


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
        patient_ids: Mapping[str, str] | None = None,
        excluded_sample_ids: Sequence[str] = (),
    ) -> None:
        self.split_dir = Path(split_dir)
        self.image_size = image_size
        excluded_ids = {str(sample_id) for sample_id in excluded_sample_ids}

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
        found_excluded_ids: set[str] = set()
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
            if sample_id in excluded_ids:
                found_excluded_ids.add(sample_id)
                continue
            class_names.add(class_name)
            parsed.append((path, class_name, sample_id, slice_index))

        if not parsed:
            raise ValueError(
                f"No usable JPEG images remain inside {self.split_dir}"
            )
        self.excluded_sample_ids = tuple(sorted(found_excluded_ids))

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

        scan_classes: dict[str, str] = {}
        for class_name, sample_id in grouped:
            previous_class = scan_classes.setdefault(sample_id, class_name)
            if previous_class != class_name:
                raise ValueError(
                    f"Scan {sample_id!r} appears in both {previous_class!r} "
                    f"and {class_name!r}"
                )

        self.samples: list[ADNISample] = []
        for (class_name, sample_id), slices in sorted(grouped.items()):
            slices.sort(key=lambda item: (item[0], str(item[1])))
            indices = tuple(index for index, _ in slices)
            if len(indices) != len(set(indices)):
                raise ValueError(
                    f"Sample {sample_id!r} in class {class_name!r} has duplicate "
                    "slice indices"
                )
            if patient_ids is None:
                patient_id = sample_id
            else:
                try:
                    patient_id = patient_ids[sample_id]
                except KeyError as error:
                    raise ValueError(
                        f"Scan {sample_id!r} is absent from the ADNI metadata"
                    ) from error
            self.samples.append(
                ADNISample(
                    sample_id=sample_id,
                    label=self.class_to_idx[class_name],
                    slice_paths=tuple(path for _, path in slices),
                    slice_indices=indices,
                    patient_id=patient_id,
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
        normalize_per_slice: bool = False,
        slice_context: int = 1,
        transform: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        self.class_to_idx = dict(subject_dataset.class_to_idx)
        self.image_size = subject_dataset.image_size
        self.excluded_sample_ids = subject_dataset.excluded_sample_ids
        self.foreground_statistics = foreground_statistics
        self.normalize_per_slice = normalize_per_slice
        self.slice_context = slice_context
        self.transform = transform

        if slice_context not in {1, 3}:
            raise ValueError("slice_context must be 1 or 3")

        if foreground_statistics is not None and normalize_per_slice:
            raise ValueError(
                "training-set and per-slice normalization cannot both be enabled"
            )

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
        if any(
            index < 0 or index >= len(subject_dataset)
            for index in selected_indices
        ):
            raise IndexError("subject index is outside the dataset")

        self.subject_ids = tuple(
            subject_dataset.samples[index].sample_id for index in selected_indices
        )
        self.patient_ids = tuple(
            subject_dataset.samples[index].patient_id for index in selected_indices
        )
        self.slices: list[ADNISlice] = []
        for subject_index in selected_indices:
            sample = subject_dataset.samples[subject_index]
            final_position = len(sample.slice_paths) - 1
            for position, (path, slice_index) in enumerate(
                zip(sample.slice_paths, sample.slice_indices)
            ):
                previous_path = sample.slice_paths[max(position - 1, 0)]
                next_path = sample.slice_paths[min(position + 1, final_position)]
                self.slices.append(
                    ADNISlice(
                        sample_id=sample.sample_id,
                        label=sample.label,
                        path=path,
                        context_paths=(previous_path, path, next_path),
                        slice_index=slice_index,
                        patient_id=sample.patient_id,
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

    @property
    def patient_count(self) -> int:
        return len(set(self.patient_ids))

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = self.slices[index]
        if self.slice_context == 3:
            # ConvNeXt still receives three channels, but they now contain
            # the previous, centre and next grayscale MRI slices instead of
            # three copies of the centre slice. Context never crosses a scan.
            image = torch.stack(
                [
                    _image_to_tensor(path, self.image_size)[0]
                    for path in item.context_paths
                ]
            )
        else:
            image = _image_to_tensor(item.path, self.image_size)
        if self.transform is not None:
            image = self.transform(image)
        foreground = image > _FOREGROUND_THRESHOLD
        if self.normalize_per_slice:
            normalized_channels = []
            for channel, channel_foreground in zip(image, foreground):
                foreground_values = channel.masked_select(channel_foreground)
                if foreground_values.numel() == 0:
                    # Completely black or constant slices contain no usable
                    # contrast and are represented by a neutral zero channel.
                    normalized_channel = torch.zeros_like(channel)
                else:
                    foreground_mean = foreground_values.mean()
                    foreground_standard_deviation = foreground_values.std(
                        unbiased=False
                    )
                    minimum_standard_deviation = torch.finfo(image.dtype).eps
                    if (
                        foreground_standard_deviation
                        <= minimum_standard_deviation
                    ):
                        normalized_channel = torch.zeros_like(channel)
                    else:
                        normalized_channel = torch.where(
                            channel_foreground,
                            (channel - foreground_mean)
                            / foreground_standard_deviation,
                            0.0,
                        )
                normalized_channels.append(normalized_channel)
            image = torch.stack(normalized_channels)
        elif self.foreground_statistics is not None:
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
            "patient_id": item.patient_id,
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


class DiagnosisPatientBalancedSampler(Sampler[int]):
    """Balance diagnoses, then patients, while retaining a full-length epoch."""

    def __init__(self, dataset: ADNISliceDataset, *, seed: int) -> None:
        grouped_indices: dict[tuple[int, str], list[int]] = defaultdict(list)
        for index, item in enumerate(dataset.slices):
            grouped_indices[(item.label, item.patient_id)].append(index)
        if not grouped_indices:
            raise ValueError("Cannot sample slices from an empty dataset")

        class_groups: dict[int, list[tuple[int, ...]]] = defaultdict(list)
        for (label, _), indices in sorted(grouped_indices.items()):
            class_groups[label].append(tuple(indices))
        if len(class_groups) < 2:
            raise ValueError("Patient-balanced sampling requires at least two classes")

        self.class_groups = tuple(
            tuple(class_groups[label]) for label in sorted(class_groups)
        )
        self.epoch_size = len(dataset)
        self.seed = seed
        self.epoch = 0

    @staticmethod
    def _sample_group(
        indices: tuple[int, ...],
        count: int,
        generator: torch.Generator,
    ) -> list[int]:
        selected: list[int] = []
        while len(selected) < count:
            order = torch.randperm(len(indices), generator=generator).tolist()
            remaining = count - len(selected)
            selected.extend(indices[index] for index in order[:remaining])
        return selected

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1

        selected_indices: list[int] = []
        class_count = len(self.class_groups)
        class_order = torch.randperm(class_count, generator=generator).tolist()
        class_targets = [self.epoch_size // class_count] * class_count
        for class_index in class_order[: self.epoch_size % class_count]:
            class_targets[class_index] += 1

        for groups, class_target in zip(self.class_groups, class_targets):
            patient_count = len(groups)
            patient_order = torch.randperm(
                patient_count,
                generator=generator,
            ).tolist()
            patient_targets = [class_target // patient_count] * patient_count
            for patient_index in patient_order[: class_target % patient_count]:
                patient_targets[patient_index] += 1
            for indices, count in zip(groups, patient_targets):
                selected_indices.extend(
                    self._sample_group(indices, count, generator)
                )

        order = torch.randperm(
            len(selected_indices),
            generator=generator,
        ).tolist()
        return iter(selected_indices[index] for index in order)

    def __len__(self) -> int:
        return self.epoch_size


def _patient_stratified_split(
    samples: Sequence[ADNISample],
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Split complete patients while preserving both diagnoses."""

    patient_indices: dict[str, list[int]] = defaultdict(list)
    patient_labels: dict[str, int] = {}
    for index, sample in enumerate(samples):
        patient_indices[sample.patient_id].append(index)
        previous_label = patient_labels.setdefault(sample.patient_id, sample.label)
        if previous_label != sample.label:
            raise ValueError(
                f"Patient {sample.patient_id!r} has both AD and NC scans"
            )

    patients_by_label: dict[int, list[str]] = defaultdict(list)
    for patient_id, label in patient_labels.items():
        patients_by_label[label].append(patient_id)

    generator = torch.Generator().manual_seed(seed)
    validation_patients: set[str] = set()
    for label in sorted(patients_by_label):
        patient_ids = sorted(patients_by_label[label])
        if len(patient_ids) < 2:
            raise ValueError(
                f"Class {label} needs at least two patients for a grouped split"
            )
        validation_count = max(1, round(len(patient_ids) * validation_fraction))
        validation_count = min(validation_count, len(patient_ids) - 1)
        permutation = torch.randperm(
            len(patient_ids),
            generator=generator,
        ).tolist()
        validation_patients.update(
            patient_ids[index] for index in permutation[:validation_count]
        )

    training_indices: list[int] = []
    validation_indices: list[int] = []
    for patient_id, indices in patient_indices.items():
        target = (
            validation_indices
            if patient_id in validation_patients
            else training_indices
        )
        target.extend(indices)
    return sorted(training_indices), sorted(validation_indices)


def create_adni_slice_dataloaders(
    data_dir: str | Path | None = None,
    *,
    batch_size: int = 48,
    validation_fraction: float = 0.2,
    seed: int = 42,
    training_seed: int | None = None,
    training_slices_per_scan: int | None = None,
    split_level: str = "scan",
    metadata_path: str | Path | None = None,
    normalization: str = "training",
    slice_context: int = 1,
    balance_patients: bool = False,
    excluded_sample_ids: Sequence[str] = (),
    image_size: tuple[int, int] | None = (224, 224),
    train_transform: Callable[[Tensor], Tensor] | None = None,
    evaluation_transform: Callable[[Tensor], Tensor] | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create slice loaders after a deterministic grouped split.

    Groups are split before their records are flattened into individual
    slices. A patient-level split additionally keeps repeat scans from one
    patient together. With three-slice context, adjacent slices are assembled
    only after splitting and always come from the centre slice's scan.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if training_slices_per_scan is not None and training_slices_per_scan <= 0:
        raise ValueError("training_slices_per_scan must be positive")
    if split_level not in {"scan", "patient"}:
        raise ValueError("split_level must be 'scan' or 'patient'")
    if normalization not in {"training", "per-slice"}:
        raise ValueError("normalization must be 'training' or 'per-slice'")
    if slice_context not in {1, 3}:
        raise ValueError("slice_context must be 1 or 3")
    if training_slices_per_scan is not None and balance_patients:
        raise ValueError(
            "slice subsampling and patient balancing cannot both be enabled"
        )

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

    patient_ids = None
    if split_level == "patient" or balance_patients:
        resolved_metadata_path = (
            data_path.parent / "meta_data_with_label.json"
            if metadata_path is None
            else Path(metadata_path)
        )
        patient_ids = _load_patient_ids(resolved_metadata_path)

    train_dir = _find_split_directory(data_path, ("train", "training"))
    test_dir = _find_split_directory(data_path, ("test", "testing"))
    training_subjects = ADNISubjectIndex(
        train_dir,
        image_size=image_size,
        patient_ids=patient_ids,
        excluded_sample_ids=excluded_sample_ids,
    )
    test_subjects = ADNISubjectIndex(
        test_dir,
        class_to_idx=training_subjects.class_to_idx,
        image_size=image_size,
        patient_ids=patient_ids,
        excluded_sample_ids=excluded_sample_ids,
    )

    training_identities = {
        sample.sample_id for sample in training_subjects.samples
    }
    test_identities = {sample.sample_id for sample in test_subjects.samples}
    if training_identities & test_identities:
        raise RuntimeError(
            "A scan appears in both the provided train and test splits"
        )
    if patient_ids is not None:
        patient_labels: dict[str, int] = {}
        for sample in (*training_subjects.samples, *test_subjects.samples):
            previous_label = patient_labels.setdefault(
                sample.patient_id,
                sample.label,
            )
            if previous_label != sample.label:
                raise ValueError(
                    f"Patient {sample.patient_id!r} has both AD and NC scans"
                )

    scan_count = len(training_subjects)
    if scan_count < 2:
        raise ValueError("At least two grouped training scans are required")
    if split_level == "patient":
        training_indices, validation_indices = _patient_stratified_split(
            training_subjects.samples,
            validation_fraction,
            seed,
        )
    else:
        validation_count = max(1, round(scan_count * validation_fraction))
        training_count = scan_count - validation_count
        if training_count == 0:
            raise ValueError("validation_fraction leaves no scans for training")
        permutation = torch.randperm(
            scan_count,
            generator=torch.Generator().manual_seed(seed),
        ).tolist()
        training_indices = permutation[:training_count]
        validation_indices = permutation[training_count:]

    foreground_statistics = None
    if normalization == "training":
        foreground_statistics = _calculate_foreground_statistics(
            training_subjects.samples,
            training_indices,
            image_size,
        )
    normalize_per_slice = normalization == "per-slice"

    training_slices = ADNISliceDataset(
        training_subjects,
        training_indices,
        foreground_statistics=foreground_statistics,
        normalize_per_slice=normalize_per_slice,
        slice_context=slice_context,
        transform=train_transform,
    )
    validation_slices = ADNISliceDataset(
        training_subjects,
        validation_indices,
        foreground_statistics=foreground_statistics,
        normalize_per_slice=normalize_per_slice,
        slice_context=slice_context,
        transform=evaluation_transform,
    )
    test_slices = ADNISliceDataset(
        test_subjects,
        foreground_statistics=foreground_statistics,
        normalize_per_slice=normalize_per_slice,
        slice_context=slice_context,
        transform=evaluation_transform,
    )

    if set(training_slices.subject_ids) & set(validation_slices.subject_ids):
        raise RuntimeError("A scan appears in both training and validation")
    if (
        split_level == "patient"
        and set(training_slices.patient_ids) & set(validation_slices.patient_ids)
    ):
        raise RuntimeError("A patient appears in both training and validation")

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
    elif balance_patients:
        training_sampler = DiagnosisPatientBalancedSampler(
            training_slices,
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
