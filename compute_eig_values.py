import os
from omegaconf import OmegaConf
import torch
import h5py
from tqdm import tqdm
from pathlib import Path
import numpy as np
from hydra import initialize_config_dir, compose
import matplotlib.pyplot as plt
import argparse

from idn.loader.loader_dsec import (
    Sequence,
    RepresentationType,
    assemble_dsec_sequences,
    assemble_dsec_test_set,
)

# Parse command-line arguments for HPC jobs
parser = argparse.ArgumentParser(description="Process dataset indices for HPC")
parser.add_argument(
    "--start_idx", type=int, default=0, help="Start index of dataset range"
)
parser.add_argument("--end_idx", type=int, default=1, help="End index of dataset range")
parser.add_argument(
    "--config_name", type=str, required=True, help="Name of the config file"
)
parser.add_argument("--data_root", type=str, default="data", help="Root directory for data")
parser.add_argument("--test", action="store_true", help="Run in test mode with a small dataset")
parser.add_argument("--test_set_root", type=str, default="data/test", help="Root directory for test set")
args = parser.parse_args()

# Conditional requirement check
if args.test and not args.test_set_root:
    parser.error("--test_set_root is required when --test is set")

# Path to the directory where your config folder is
config_dir = os.path.abspath("idn/config")  # or give full path

# Optional: print to confirm
print("Loading configs from:", config_dir)

with initialize_config_dir(config_dir=config_dir, job_name="notebook_job"):
    cfg = compose(config_name=args.config_name)  # the YAML file id_train.yaml

cfg.dataset.train.force_preprocess = True
cfg.dataset.train.in_memory = False
cfg.dataset.train.do_not_save_preprocessed = False
cfg.dataset.train.add_eigenvalues = True
cfg.dataset.train.add_filter_values = True

print(OmegaConf.to_yaml(cfg))

if args.test:
    print(f"Using test set root: {args.test_set_root}")
    dataset = assemble_dsec_test_set(
        args.test_set_root,
        seq_len=None,
        concat_seq=True,
        config=cfg.dataset.train,
        representation_type=cfg.dataset.get("representation_type", None),
    )
else:
    train_seqs = [
        val_seq
        for x in cfg.get("validation", dict()).values()
        for val_seq in x.dataset.train.seq
    ]
    if train_seqs:
        include_seq = train_seqs + [
            val_seq
            for x in cfg.get("validation", dict()).values()
            for val_seq in x.dataset.val.seq
        ]
    else:
        include_seq = []

    dataset = assemble_dsec_sequences(
        args.data_root,
        include_seq=include_seq,
        exclude_seq=None,
        require_gt=True,
        config=cfg.dataset.train,
        representation_type=cfg.dataset.get("representation_type", None),
        num_bins=cfg.dataset.get("num_voxel_bins", None),
    )

datasets_len = len(dataset)

print(f"The length of dataset: {datasets_len}")

start_idx = args.start_idx
end_idx = args.end_idx

print(f"Processing dataset indices from {start_idx} t/m {end_idx - 1}")

assert 0 <= start_idx < datasets_len, "Start index out of bounds"
assert 0 < end_idx <= datasets_len, "End index out of bounds"
assert start_idx < end_idx, "Start index must be less than end index"

assert cfg.dataset.train.get("add_eigenvalues", False) or cfg.dataset.train.get("add_filter_values", False), \
    "You need to set add_eigenvalues or add_filter_values to True in your config file."
add_eigenvalues = cfg.dataset.train.get("add_eigenvalues", False)
add_filter_values = cfg.dataset.train.get("add_filter_values", False)
print(f"add_eigenvalues: {add_eigenvalues}, add_filter_values: {add_filter_values}")

# Process each item in the specified range
for i in tqdm(range(start_idx, end_idx), desc="Processing samples"):
    try:
        sample = dataset[i]
        # Do your processing here
        if i % 10 == 0:
            dataset_type_dir = "test" if args.test else "train_events"
            seq_path = (
                Path(args.data_root) / dataset_type_dir / sample["seq_name"]
            )
            event_voxel_path = seq_path / "event_voxel"
            if not event_voxel_path.exists():
                event_voxel_path.mkdir(parents=True, exist_ok=True)
            if add_eigenvalues:
                eig_1_path = seq_path / "eig_1"
                if not eig_1_path.exists():
                    eig_1_path.mkdir(parents=True, exist_ok=True)
                eig_2_path = seq_path / "eig_2"
                if not eig_2_path.exists():
                    eig_2_path.mkdir(parents=True, exist_ok=True)
            if add_filter_values:
                filter_value_path = seq_path / "filter_values"
                if not filter_value_path.exists():
                    filter_value_path.mkdir(parents=True, exist_ok=True)
                
            for channel in range(0, 15, 5):
                file_path = (
                    event_voxel_path
                    / f"event_voxel_{sample['file_index']}_{channel}.png"
                )
                fig, ax = plt.subplots(1, 1, figsize=(10, 7))
                im = ax.imshow(sample["event_volume_new"][channel, :, :], cmap="binary")
                ax.set_title(f"Augmented event voxel channel {channel}")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                plt.tight_layout()
                plt.savefig(file_path)
                plt.close()
                if add_eigenvalues:
                    file_path = eig_1_path / f"eig_1_{sample['file_index']}_{channel}.png"
                    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
                    im = ax.imshow(
                        sample["eigenvalues_volume_new"][channel, :, :], cmap="binary"
                    )
                    ax.set_title(f"Augmented eig value 1 channel {channel}")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    plt.tight_layout()
                    plt.savefig(file_path)
                    plt.close()

                    file_path = eig_2_path / f"eig_2_{sample['file_index']}_{channel}.png"
                    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
                    im = ax.imshow(
                        sample["eigenvalues_volume_new"][channel + 15, :, :], cmap="binary"
                    )
                    ax.set_title(f"Augmented eig value 2 channel {channel}")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    plt.tight_layout()
                    plt.savefig(file_path)
                    plt.close()
                if add_filter_values:
                    file_path = filter_value_path / f"filter_values_{sample['file_index']}_{channel}.png"
                    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
                    im = ax.imshow(
                        sample["filter_values_volume_new"][channel, :, :], cmap="binary"
                    )
                    ax.set_title(f"Augmented filter values channel {channel}")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    plt.tight_layout()
                    plt.savefig(file_path)
                    plt.close()
        print(f"Index {i} processed.")

    except Exception as e:
        print(f"Error processing index {i}: {e}")
        raise Exception("Training failed")
