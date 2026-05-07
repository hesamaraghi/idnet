from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from idn.loader.loader_dsec import HarrisRecursive
from idn.utils.mvsec_utils import EventSequence
from idn.utils.transformers import EventSequenceToVoxelGrid_Pytorch


class EVIMOv2Sequence(Dataset):
    def __init__(self, seq_path, config, split="eval", num_bins=None):
        self.seq_path = Path(seq_path)
        self.config = config
        self.split = split
        self.seq_name = self.seq_path.name

        self.num_bins = int(num_bins or config.get("num_voxel_bins", 15))
        self.add_eigenvalues = config.get("add_eigenvalues", False)
        self.add_filter_values = config.get("add_filter_values", False)
        self.in_memory = config.get("in_memory", False)
        self.force_preprocess = config.get("force_preprocess", False)
        self.do_not_save_preprocessed = config.get("do_not_save_preprocessed", True)
        self.normalize_voxel = config.get("normalize_voxel", True)
        self.skip_invalid = config.get("skip_invalid", True)

        self.image_width = int(config.get("image_width", 640))
        self.image_height = int(config.get("image_height", 480))

        self.events_xy = np.load(self.seq_path / "dataset_events_xy.npy", mmap_mode="r")
        self.events_t = np.load(self.seq_path / "dataset_events_t.npy", mmap_mode="r")
        self.events_p = np.load(self.seq_path / "dataset_events_p.npy", mmap_mode="r")
        self.flow_data = np.load(self.seq_path / "dataset_flow.npz")
        self.mask_data = np.load(self.seq_path / "dataset_mask.npz")

        self.voxel = EventSequenceToVoxelGrid_Pytorch(
            num_bins=self.num_bins,
            normalize=self.normalize_voxel,
            gpu=False,
        )
        self.voxel_not_normalized = EventSequenceToVoxelGrid_Pytorch(
            num_bins=self.num_bins,
            normalize=False,
            gpu=False,
        )

        if self.add_eigenvalues or self.add_filter_values:
            self.tau = config.get("tau", 15_000)
            self.filter_size = config.get("filter_size", 7)
            self.harris_recursive = HarrisRecursive(
                tau=self.tau,
                filter_size=self.filter_size,
                image_size=(self.image_height, self.image_width),
            )

        self.samples = self._build_sample_index()
        self.preprocessed_path = self._build_preprocessed_path()

        if self.in_memory:
            self.data = [self.get_data_sample(idx) for idx in range(len(self))]

    def _build_preprocessed_path(self):
        preprocessed_root = self.config.common.get("preprocessed_root", None)
        if preprocessed_root is None:
            preprocessed_root = Path(self.config.common.data_root) / "preprocessed"
        tag = (
            f"bins{self.num_bins}_eig{int(self.add_eigenvalues)}"
            f"_filter{int(self.add_filter_values)}"
        )
        if self.add_eigenvalues or self.add_filter_values:
            tag += f"_tau{self.tau}_fs{self.filter_size}"
        path = Path(preprocessed_root) / self.split / self.seq_name / tag
        if not self.do_not_save_preprocessed:
            path.mkdir(parents=True, exist_ok=True)
        return path

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
            raise ValueError(f"{self.seq_name}: no valid EVIMOv2 flow samples found")
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
            return self.data[idx]
        return self.get_data_sample(idx)

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
            return torch.load(preprocessed_file_path, weights_only=False)

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
        return self.voxel_not_normalized(
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
    if not split_root.exists():
        raise FileNotFoundError(f"EVIMOv2 split directory does not exist: {split_root}")

    available_seqs = sorted(
        path.name for path in split_root.iterdir() if path.is_dir()
    )
    if include_seq:
        include_seq = [include_seq] if isinstance(include_seq, str) else list(include_seq)
        seqs = [seq for seq in available_seqs if seq in include_seq]
    else:
        seqs = available_seqs
    if not seqs:
        raise ValueError(f"No EVIMOv2 sequences selected from {split_root}")

    return [
        EVIMOv2Sequence(split_root / seq, config=config, split=split, num_bins=num_bins)
        for seq in seqs
    ]


__all__ = ["EVIMOv2Sequence", "assemble_evimo_sequences"]
