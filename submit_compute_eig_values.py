import os
from omegaconf import OmegaConf
import torch
import h5py
from tqdm import tqdm
from pathlib import Path
import numpy as np
from hydra import initialize_config_dir, compose
import subprocess
import argparse
import warnings

from idn.loader.loader_dsec import (
    Sequence,
    RepresentationType,
    assemble_dsec_sequences,
    assemble_dsec_test_set,
)

def submit_jobs(data_root, chunk_size, submit_script, python_script, config_name, test, test_set_root):

    # Path to the directory where your config folder is
    config_dir = os.path.abspath("idn/config")  # or give full path

    # Optional: print to confirm
    print("Loading configs from:", config_dir)

    with initialize_config_dir(config_dir=config_dir, job_name="notebook_job"):
        cfg = compose(config_name=config_name)  # the YAML file id_train.yaml

    cfg.dataset.train.force_preprocess = True
    cfg.dataset.train.in_memory = False

    print(OmegaConf.to_yaml(cfg))

    if test:
        dataset = assemble_dsec_test_set(
            test_set_root,
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

        if not os.path.exists(data_root):
            warnings.warn(
                f"Data root {data_root} does not exist. "
                "Using default root in data/ instead."
            )
            data_root = "data"
        dataset = assemble_dsec_sequences(
            data_root,
            include_seq= include_seq,
            exclude_seq=None,
            require_gt=True,
            config=cfg.dataset.train,
            representation_type=cfg.dataset.get("representation_type", None),
            num_bins=cfg.dataset.get("num_voxel_bins", None),
        )

    dataset_size = len(dataset)

    print(f"The length of dataset: {dataset_size}")

    for start_idx in range(0, dataset_size, chunk_size):
        end_idx = min(start_idx + chunk_size, dataset_size)

        # Full Python command to pass to the sbatch script
        sbatch_command = (
            f"sbatch {submit_script} "
            f"python {python_script} "
            f"--data_root {data_root} "
            f"--start_idx {str(start_idx)} "
            f"--end_idx {str(end_idx)} "
            f"--config_name {config_name} "
        ) + \
            (f"--test --test_set_root {test_set_root} " if test else "")

        print(f"Submitting job for indices {start_idx} to {end_idx}: {sbatch_command}")
        subprocess.call(sbatch_command, shell=True)
        # break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit dataset jobs using SLURM.")
    parser.add_argument(
        "--chunk_size", type=int, required=True, help="Number of samples per job"
    )
    parser.add_argument(
        "--submit_script", type=str, required=True, help="Path to submit_job.sh"
    )
    parser.add_argument(
        "--python_script",
        type=str,
        default="compute_eig_values.py",
        help="Path to the processing Python script",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        required=True,
        help="Name of the configuration file without the .yaml extension",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with a small dataset",
    )
    parser.add_argument(
        "--data_root", type=str, default="data", help="Root directory for data"
    )
    parser.add_argument(
        "--test_set_root",
        type=str,
        default="data/test",
        help="Root directory for test set",
    )
    args = parser.parse_args()

    # Conditional requirement check
    if args.test and not args.test_set_root:
        parser.error("--test_set_root is required when --test is set")

    submit_jobs(
        data_root=args.data_root,
        chunk_size=args.chunk_size,
        submit_script=args.submit_script,
        python_script=args.python_script,
        config_name=args.config_name,
        test=args.test,
        test_set_root=args.test_set_root,
    )
