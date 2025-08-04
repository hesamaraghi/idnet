import os
from omegaconf import OmegaConf
import torch
import h5py
from tqdm import tqdm
from pathlib import Path
import numpy as np
from hydra import initialize_config_dir, compose
import matplotlib.pyplot as plt


# Path to the directory where your config folder is
config_dir = os.path.abspath("idn/config")  # or give full path

# Optional: print to confirm
print("Loading configs from:", config_dir)

with initialize_config_dir(config_dir=config_dir, job_name="notebook_job"):
    cfg = compose(config_name="id_train_eigen")  # the YAML file id_train.yaml

cfg.dataset.train.force_preprocess = True
cfg.dataset.train.in_memory = False
cfg.dataset.train.do_not_save_preprocessed = True

cfg.dataset.train.add_eigenvalues = True
cfg.dataset.train.add_filter_values = True
cfg.dataset.train.tau = 15_000 
cfg.dataset.train.filter_size = 7

cfg.dataset.train.vertical_flip = False
cfg.dataset.train.horizontal_flip = False
cfg.dataset.train.random_crop = False

import argparse

# Parse command-line arguments for HPC jobs
parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument(
    "--modified",
    help="Use modified voxel grid representation (default: False)",
    action="store_true",
)

modified_voxel_grid = parser.parse_args().modified

print(f"Using modified voxel grid representation: {modified_voxel_grid}")

if modified_voxel_grid:
    from testing_loader_dsec import (
        Sequence,
        RepresentationType,
        assemble_dsec_sequences,
    )

    save_dir = Path("saved_images") / "modified_voxel_grid_representation"
else:
    from idn.loader.loader_dsec import (
        Sequence,
        RepresentationType,
        assemble_dsec_sequences,
    )

    save_dir = Path("saved_images") / "representation"
if not save_dir.exists():
    save_dir.mkdir(parents=True, exist_ok=True)

# print(OmegaConf.to_yaml(cfg))
taus = [1_000, 5_000, 15_000]
filter_sizes = [3, 5, 7]

for tau in taus:
    for filter_size in filter_sizes:
        
        cfg.dataset.train.tau = tau 
        cfg.dataset.train.filter_size = filter_size

        dataset = assemble_dsec_sequences(
            cfg.dataset.common.data_root,
            include_seq=set(
                [
                    val_seq
                    for x in cfg.get("validation", dict()).values()
                    for val_seq in x.dataset.train.seq
                ]
                + [
                    val_seq
                    for x in cfg.get("validation", dict()).values()
                    for val_seq in x.dataset.val.seq
                ]
            ),
            exclude_seq=None,
            require_gt=True,
            config=cfg.dataset.train,
            representation_type=cfg.dataset.get("representation_type", None),
            num_bins=cfg.dataset.get("num_voxel_bins", None),
        )

        add_eigenvalues = cfg.dataset.train.get("add_eigenvalues", False)
        add_filter_values = cfg.dataset.train.get("add_filter_values", False)
        print(f"add_eigenvalues: {add_eigenvalues}, add_filter_values: {add_filter_values}")

        for index in [0, 500, 1000]:

            sample = dataset[index]
            seq_path =  save_dir

            event_voxel_path = seq_path / "event_voxel"
            if not event_voxel_path.exists():
                event_voxel_path.mkdir(parents=True, exist_ok=True)
            if add_eigenvalues:
                eig1_path = seq_path / "eig1"
                if not eig1_path.exists():
                    eig1_path.mkdir(parents=True, exist_ok=True)
                eig2_path = seq_path / "eig_2"
                if not eig2_path.exists():
                    eig2_path.mkdir(parents=True, exist_ok=True)
            if add_filter_values:
                filter_value_path = seq_path / "filter_values"
                if not filter_value_path.exists():
                    filter_value_path.mkdir(parents=True, exist_ok=True)

            for channel in range(0, 15, 3):
                file_path = (
                    event_voxel_path
                    / f"event_voxel_{sample['seq_name']}_{sample['file_index']}_{channel}.png"
                )
                fig, ax = plt.subplots(1, 1, figsize=(10, 7))
                im = ax.imshow(sample["event_volume_new"][channel, :, :], cmap="binary")
                ax.set_title(f"event voxel channel {channel}")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                plt.tight_layout()
                plt.savefig(file_path)
                plt.close()
                if add_eigenvalues:
                    file_path = eig1_path / f"eig1_{sample['seq_name']}_{sample['file_index']}_{channel}_tau_{cfg.dataset.train.tau}_filtersize_{cfg.dataset.train.filter_size}.png"
                    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
                    im = ax.imshow(
                        sample["eigenvalues_volume_new"][channel, :, :], cmap="binary"
                    )
                    ax.set_title(f"eig1 channel {channel}: tau {cfg.dataset.train.tau}, filter size {cfg.dataset.train.filter_size}")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    plt.tight_layout()
                    plt.savefig(file_path)
                    plt.close()

                    file_path = eig2_path / f"eig2_{sample['seq_name']}_{sample['file_index']}_{channel}_tau_{cfg.dataset.train.tau}_filtersize_{cfg.dataset.train.filter_size}.png"
                    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
                    im = ax.imshow(
                        sample["eigenvalues_volume_new"][channel + 15, :, :], cmap="binary"
                    )
                    ax.set_title(f"eig2 channel {channel}: tau {cfg.dataset.train.tau}, filter size {cfg.dataset.train.filter_size}")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    plt.tight_layout()
                    plt.savefig(file_path)
                    plt.close()
                if add_filter_values:
                    file_path = filter_value_path / f"filter_values_{sample['seq_name']}_{sample['file_index']}_{channel}_tau_{cfg.dataset.train.tau}_filtersize_{cfg.dataset.train.filter_size}.png"
                    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
                    im = ax.imshow(
                        sample["filter_values_volume_new"][channel, :, :], cmap="binary"
                    )
                    ax.set_title(f"filter values channel {channel}: tau {cfg.dataset.train.tau}, filter size {cfg.dataset.train.filter_size}")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    plt.tight_layout()
                    plt.savefig(file_path)
                    plt.close()
