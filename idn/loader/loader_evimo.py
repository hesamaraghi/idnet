from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from idn.loader.loader_dsec import HarrisRecursive
from idn.utils.mvsec_utils import EventSequence
from idn.utils.transformers import (
    EventSequenceToVoxelGrid_Pytorch,
    apply_randomcrop_to_sample,
    apply_transform_to_field,
    downsample_spatial,
    downsample_spatial_mask,
)


def _section_for_split(split):
    if split == "train":
        return "train"
    if split in ("eval", "test", "val"):
        return "val"
    return split


def _get_split_config(config, split):
    section_name = _section_for_split(split)
    return config.get(section_name, None) if config is not None else None


def _get_split_value(config, split, key, default=None):
    split_config = _get_split_config(config, split)
    if split_config is not None and key in split_config:
        value = split_config.get(key)
        if value is not None:
            return value
    return config.get(key, default) if config is not None else default


def _build_transforms(config, split):
    transforms = dict()
    downsample_ratio = _get_split_value(config, split, "downsample_ratio", 1)
    if downsample_ratio is not None and downsample_ratio > 1:
        transforms["(?<!flow_gt_)event_volume"] = lambda sample: downsample_spatial(
            sample, downsample_ratio
        )
        transforms["flow_gt"] = lambda sample: [
            downsample_spatial(sample[0], downsample_ratio) / downsample_ratio,
            downsample_spatial_mask(sample[1], downsample_ratio),
        ]
    if _get_split_value(config, split, "horizontal_flip", None):
        transforms["hflip"] = None
    if _get_split_value(config, split, "vertical_flip", None):
        transforms["vflip"] = _get_split_value(config, split, "vertical_flip")
    random_crop = _get_split_value(config, split, "random_crop", None)
    if random_crop:
        transforms["randomcrop"] = random_crop
    return transforms


def _get_preprocessed_split_root(config, split):
    if config is None or "common" not in config:
        return None
    preprocessed_root = config.common.get("preprocessed_root", None)
    if preprocessed_root is None:
        return None
    return Path(preprocessed_root) / split


def _format_cache_tag_value(value):
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


class EVIMO2v2Sequence(Dataset):
    def __init__(self, seq_path, config, split="eval", num_bins=None, transforms=None):
        self.seq_path = Path(seq_path)
        self.config = config
        self.split = split
        self.seq_name = self.seq_path.name
        self.transforms = transforms or dict()

        self.num_bins = int(num_bins or _get_split_value(config, split, "num_voxel_bins", 15))
        self.add_eigenvalues = _get_split_value(config, split, "add_eigenvalues", False)
        self.add_filter_values = _get_split_value(config, split, "add_filter_values", False)
        self.tau = _get_split_value(config, split, "tau", 15_000)
        self.filter_size = _get_split_value(config, split, "filter_size", 7)
        self.in_memory = _get_split_value(config, split, "in_memory", False)
        self.force_preprocess = _get_split_value(config, split, "force_preprocess", False)
        self.do_not_save_preprocessed = _get_split_value(
            config, split, "do_not_save_preprocessed", True
        )
        self.normalize_voxel = _get_split_value(config, split, "normalize_voxel", True)
        self.normalize_aux_voxel = _get_split_value(config, split, "normalize_aux_voxel", True)
        self.skip_invalid = _get_split_value(config, split, "skip_invalid", True)

        self.image_width = int(_get_split_value(config, split, "image_width", 640))
        self.image_height = int(_get_split_value(config, split, "image_height", 480))

        self.preprocessed_path = self._build_preprocessed_path()
        self.samples = self._build_cached_sample_index()
        self.using_preprocessed_cache = bool(self.samples)

        self.events_xy = None
        self.events_t = None
        self.events_p = None
        self.flow_data = None
        self.mask_data = None
        self.voxel = None
        self.voxel_aux = None
        self.harris_recursive = None

        if not self.using_preprocessed_cache:
            self._open_data_files()
            self.voxel = EventSequenceToVoxelGrid_Pytorch(
                num_bins=self.num_bins,
                normalize=self.normalize_voxel,
                gpu=False,
            )
            self.voxel_aux = EventSequenceToVoxelGrid_Pytorch(
                num_bins=self.num_bins,
                normalize=self.normalize_aux_voxel,
                gpu=False,
            )
            if self.add_eigenvalues or self.add_filter_values:
                self.harris_recursive = HarrisRecursive(
                    tau=self.tau,
                    filter_size=self.filter_size,
                    image_size=(self.image_height, self.image_width),
                )
            self.samples = self._build_sample_index()

        if self.in_memory:
            self.data = [self.get_data_sample(idx) for idx in range(len(self))]

    def _open_data_files(self):
        self.events_xy = np.load(self.seq_path / "dataset_events_xy.npy", mmap_mode="r")
        self.events_t = np.load(self.seq_path / "dataset_events_t.npy", mmap_mode="r")
        self.events_p = np.load(self.seq_path / "dataset_events_p.npy", mmap_mode="r")
        self.flow_data = np.load(self.seq_path / "dataset_flow.npz")
        self.mask_data = np.load(self.seq_path / "dataset_mask.npz")

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ("events_xy", "events_t", "events_p", "flow_data", "mask_data"):
            state[key] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not self.using_preprocessed_cache:
            self._open_data_files()

    def _preprocessed_tag(self, add_eigenvalues, add_filter_values):
        add_eigenvalues = bool(add_eigenvalues)
        add_filter_values = bool(add_filter_values)
        tag = (
            f"bins{self.num_bins}_eig{int(add_eigenvalues)}"
            f"_filter{int(add_filter_values)}"
        )
        if add_eigenvalues or add_filter_values:
            tag += (
                f"_tau{_format_cache_tag_value(self.tau)}"
                f"_fs{_format_cache_tag_value(self.filter_size)}"
            )
        return tag

    def _build_preprocessed_path(self, add_eigenvalues=None, add_filter_values=None):
        preprocessed_root = self.config.common.get("preprocessed_root", None)
        if preprocessed_root is None:
            preprocessed_root = Path(self.config.common.data_root) / "preprocessed"
        if add_eigenvalues is None:
            add_eigenvalues = self.add_eigenvalues
        if add_filter_values is None:
            add_filter_values = self.add_filter_values
        tag = self._preprocessed_tag(add_eigenvalues, add_filter_values)
        return Path(preprocessed_root) / self.split / self.seq_name / tag

    def _candidate_preprocessed_paths(self):
        paths = [self._build_preprocessed_path()]
        rich_cache_path = self._build_preprocessed_path(
            add_eigenvalues=True,
            add_filter_values=True,
        )
        if rich_cache_path not in paths:
            paths.append(rich_cache_path)
        return paths

    def _build_cached_sample_index(self):
        if self.force_preprocess or self.do_not_save_preprocessed:
            return []
        requested_preprocessed_path = self.preprocessed_path
        for preprocessed_path in self._candidate_preprocessed_paths():
            if not preprocessed_path.exists():
                continue
            samples = []
            for slot, path in enumerate(sorted(preprocessed_path.glob("*.pt"))):
                try:
                    file_index = int(path.stem)
                except ValueError:
                    continue
                samples.append({"slot": slot, "file_index": file_index})
            if samples:
                self.preprocessed_path = preprocessed_path
                return samples
        self.preprocessed_path = requested_preprocessed_path
        return []

    def _adapt_cached_sample(self, sample, path):
        if self.add_eigenvalues and "eigenvalues_volume_new" not in sample:
            raise KeyError(f"Missing eigenvalues_volume_new in preprocessed sample {path}")
        if self.add_filter_values and "filter_values_volume_new" not in sample:
            raise KeyError(f"Missing filter_values_volume_new in preprocessed sample {path}")

        if not self.add_eigenvalues:
            sample.pop("eigenvalues_volume_new", None)
            sample.pop("eigenvalues_volume_old", None)
        if not self.add_filter_values:
            sample.pop("filter_values_volume_new", None)
            sample.pop("filter_values_volume_old", None)
        if not self.add_eigenvalues and not self.add_filter_values:
            event_volume = sample.get("event_volume_new", None)
            if torch.is_tensor(event_volume) and event_volume.shape[0] > self.num_bins:
                sample["event_volume_new"] = event_volume[: self.num_bins, :, :]
        return sample

    def _build_sample_index(self):
        flow_keys = sorted(k for k in self.flow_data.files if k.startswith("flow_"))
        t = self.flow_data["t"]
        t_end = self.flow_data["t_end"]
        if len(flow_keys) != len(t) or len(flow_keys) != len(t_end):
            raise ValueError(
                f"{self.seq_name}: expected matching flow/t/t_end counts, got "
                f"{len(flow_keys)}, {len(t)}, {len(t_end)}"
            )

        samples = []
        for i, flow_key in enumerate(flow_keys):
            file_index = int(flow_key.split("_")[-1])
            mask_key = f"mask_{file_index:010d}"
            if self.skip_invalid and not self._has_valid_flow(flow_key, mask_key):
                continue
            samples.append(
                {
                    "slot": i,
                    "flow_key": flow_key,
                    "mask_key": mask_key,
                    "file_index": file_index,
                    "t_start": float(t[i]),
                    "t_end": float(t_end[i]),
                }
            )
        if not samples:
            raise ValueError(f"{self.seq_name}: no valid EVIMO2v2 flow samples found")
        return samples

    def _has_valid_flow(self, flow_key, mask_key):
        flow = self.flow_data[flow_key]
        valid = np.isfinite(flow[..., 0]) & np.isfinite(flow[..., 1])
        if mask_key in self.mask_data.files:
            valid &= self.mask_data[mask_key] > 0
        return bool(valid.any())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.in_memory:
            sample = self.data[idx]
        else:
            sample = self.get_data_sample(idx)

        for key_t, transform in self.transforms.items():
            if key_t == "hflip":
                if random.random() > 0.5:
                    for key in sample:
                        if isinstance(sample[key], torch.Tensor):
                            sample[key] = TF.hflip(sample[key])
                        if key.startswith("flow_gt"):
                            sample[key] = [TF.hflip(mask) for mask in sample[key]]
                            sample[key][0][0, :] = -sample[key][0][0, :]
            elif key_t == "vflip":
                if random.random() < transform:
                    for key in sample:
                        if isinstance(sample[key], torch.Tensor):
                            sample[key] = TF.vflip(sample[key])
                        if key.startswith("flow_gt"):
                            sample[key] = [TF.vflip(mask) for mask in sample[key]]
                            sample[key][0][1, :] = -sample[key][0][1, :]
            elif key_t == "randomcrop":
                apply_randomcrop_to_sample(sample, crop_size=transform)
            else:
                apply_transform_to_field(sample, transform, key_t)

        return sample

    def _preprocessed_file_path(self, sample):
        return self.preprocessed_path / f"{sample['file_index']:010d}.pt"

    def get_data_sample(self, idx):
        sample_info = self.samples[idx]
        preprocessed_file_path = self._preprocessed_file_path(sample_info)
        if (
            not self.force_preprocess
            and not self.do_not_save_preprocessed
            and preprocessed_file_path.exists()
        ):
            try:
                loaded_file = torch.load(preprocessed_file_path, weights_only=False)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load EVIMO2v2 preprocessed sample: {preprocessed_file_path}"
                ) from exc
            return self._adapt_cached_sample(loaded_file, preprocessed_file_path)

        events = self._load_events(sample_info["t_start"], sample_info["t_end"])
        output = {
            "event_volume_new": self._events_to_voxel(events),
            "flow_gt_event_volume_new": self._load_flow_gt(sample_info),
            "file_index": sample_info["file_index"],
            "timestamp": sample_info["t_start"],
            "seq_name": self.seq_name,
        }

        if self.add_eigenvalues or self.add_filter_values:
            features = self._compute_features(events)
            if self.add_eigenvalues:
                eig1 = self._values_to_voxel(events, features["eig1"])
                eig2 = self._values_to_voxel(events, features["eig2"])
                output["eigenvalues_volume_new"] = torch.cat((eig1, eig2), dim=0)
            if self.add_filter_values:
                output["filter_values_volume_new"] = self._values_to_voxel(
                    events, features["filter_values"]
                )

        if not self.do_not_save_preprocessed:
            self.preprocessed_path.mkdir(parents=True, exist_ok=True)
            torch.save(output, preprocessed_file_path)
        return output

    def _load_events(self, t_start, t_end):
        start = int(np.searchsorted(self.events_t, t_start, side="left"))
        end = int(np.searchsorted(self.events_t, t_end, side="left"))
        xy = np.asarray(self.events_xy[start:end])
        return {
            "t": np.asarray(self.events_t[start:end], dtype=np.float64),
            "x": xy[:, 0].astype(np.uint16, copy=False),
            "y": xy[:, 1].astype(np.uint16, copy=False),
            "p": np.asarray(self.events_p[start:end], dtype=np.uint8),
        }

    def _events_to_voxel(self, events):
        if events["t"].size == 0:
            return torch.zeros(
                self.num_bins, self.image_height, self.image_width, dtype=torch.float32
            )
        event_array = np.column_stack(
            (events["t"], events["x"], events["y"], events["p"])
        ).astype(np.float64, copy=False)
        return self.voxel(
            EventSequence(
                None,
                params={"width": self.image_width, "height": self.image_height},
                features=event_array.copy(),
            )
        )

    def _compute_features(self, events):
        if events["t"].size == 0:
            empty = np.zeros(0, dtype=np.float32)
            return {"eig1": empty, "eig2": empty, "filter_values": empty}

        dtype = [
            ("x", np.uint16),
            ("y", np.uint16),
            ("t", np.uint64),
            ("p", np.uint8),
        ]
        harris_events = np.empty(events["t"].shape[0], dtype=dtype)
        harris_events["x"] = events["x"].astype(np.uint16)
        harris_events["y"] = events["y"].astype(np.uint16)
        harris_events["t"] = ((events["t"] - events["t"].min()) * 1e6).astype(np.uint64)
        harris_events["p"] = (events["p"] > 0).astype(np.uint8)
        self.harris_recursive(harris_events)
        return {
            "eig1": self.harris_recursive.eig1,
            "eig2": self.harris_recursive.eig2,
            "filter_values": self.harris_recursive.filter_value_recursive,
        }

    def _values_to_voxel(self, events, values):
        if events["t"].size == 0:
            return torch.zeros(
                self.num_bins, self.image_height, self.image_width, dtype=torch.float32
            )
        event_array = np.column_stack(
            (events["t"], events["x"], events["y"], values)
        ).astype(np.float64, copy=False)
        return self.voxel_aux(
            EventSequence(
                None,
                params={"width": self.image_width, "height": self.image_height},
                features=event_array.copy(),
            ),
            val_type="value",
        )

    def _load_flow_gt(self, sample):
        flow = self.flow_data[sample["flow_key"]].astype(np.float32)
        mask_key = sample["mask_key"]
        valid_mask = np.isfinite(flow[..., 0]) & np.isfinite(flow[..., 1])
        if mask_key in self.mask_data.files:
            valid_mask &= self.mask_data[mask_key] > 0
        flow = np.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)
        flow_tensor = torch.from_numpy(np.moveaxis(flow, -1, 0)).float()
        mask_tensor = torch.from_numpy(valid_mask[None, ...]).bool()
        return flow_tensor, mask_tensor


def assemble_evimo_sequences(dataset_root, split="eval", include_seq=None, config=None, num_bins=None):
    dataset_root = Path(dataset_root)
    split_root = dataset_root / split
    preprocessed_split_root = _get_preprocessed_split_root(config, split)

    if split_root.exists():
        available_seqs = sorted(
            path.name for path in split_root.iterdir() if path.is_dir()
        )
    elif preprocessed_split_root is not None and preprocessed_split_root.exists():
        available_seqs = sorted(
            path.name for path in preprocessed_split_root.iterdir() if path.is_dir()
        )
    else:
        locations = [str(split_root)]
        if preprocessed_split_root is not None:
            locations.append(str(preprocessed_split_root))
        raise FileNotFoundError(
            "EVIMO2v2 split directory does not exist in raw or preprocessed roots: "
            + ", ".join(locations)
        )
    if include_seq:
        include_seq = [include_seq] if isinstance(include_seq, str) else list(include_seq)
        seqs = [seq for seq in include_seq if seq in available_seqs]
        missing = sorted(set(include_seq) - set(seqs))
        if missing:
            raise ValueError(
                f"Requested EVIMO2v2 sequences are not available for split '{split}': {missing}"
            )
    else:
        seqs = available_seqs
    if not seqs:
        raise ValueError(f"No EVIMO2v2 sequences selected from {split_root}")

    transforms = _build_transforms(config, split)
    return [
        EVIMO2v2Sequence(
            split_root / seq,
            config=config,
            split=split,
            num_bins=num_bins,
            transforms=transforms,
        )
        for seq in seqs
    ]


__all__ = ["EVIMO2v2Sequence", "assemble_evimo_sequences"]
