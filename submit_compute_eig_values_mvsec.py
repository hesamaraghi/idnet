import os
from omegaconf import OmegaConf
from pathlib import Path
from hydra import initialize_config_dir, compose
import subprocess
import argparse
import shlex

from idn.loader.loader_mvsec import MVSEC


def configure_mvsec_dataset(
    cfg, data_root, test_set_root, num_voxel_bins=None, normalize_aux_voxel=True
):
    cfg.dataset.force_preprocess = True
    cfg.dataset.in_memory = False
    cfg.dataset.do_not_save_preprocessed = False
    cfg.dataset.add_eigenvalues = True
    cfg.dataset.add_filter_values = True
    cfg.dataset.normalize_aux_voxel = normalize_aux_voxel
    cfg.dataset.common.data_root = data_root
    cfg.dataset.common.test_root = test_set_root
    if num_voxel_bins is not None:
        cfg.dataset.num_voxel_bins = num_voxel_bins


def submit_jobs(
    data_root,
    chunk_size,
    submit_script,
    python_script,
    config_name,
    test,
    test_set_root,
    num_voxel_bins=None,
    normalize_aux_voxel=True,
    dry_run=False,
    submit_once=False,
):

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if not Path(submit_script).is_file():
        raise FileNotFoundError(f"Submit script does not exist: {submit_script}")
    if not Path(python_script).is_file():
        raise FileNotFoundError(f"Python worker script does not exist: {python_script}")

    # Path to the directory where your config folder is
    config_dir = os.path.abspath("idn/config")  # or give full path

    # Optional: print to confirm
    print("Loading configs from:", config_dir)

    with initialize_config_dir(config_dir=config_dir, job_name="notebook_job"):
        cfg = compose(config_name=config_name)  # the YAML file id_train.yaml

    configure_mvsec_dataset(
        cfg=cfg,
        data_root=data_root,
        test_set_root=test_set_root,
        num_voxel_bins=num_voxel_bins,
        normalize_aux_voxel=normalize_aux_voxel,
    )

    print(OmegaConf.to_yaml(cfg))
    
    if test:
        if "mvsec" not in cfg.validation:
            raise KeyError(
                f"Config '{config_name}' does not define validation.mvsec. "
                "Use mvsec_train_all or update the validation package name."
            )
        cfg.validation.mvsec.dataset.common.data_root = test_set_root
        cfg.validation.mvsec.dataset.common.test_root = test_set_root
        cfg.validation.mvsec.dataset.normalize_aux_voxel = normalize_aux_voxel
        dataset = MVSEC(
            config=cfg.validation.mvsec.dataset,
            training=False,
            filter=(4356, 4706),
            augment=False,
        )
    else: 
        dataset = MVSEC(
            config=cfg.dataset,
            training=True,
        )

    dataset_size = len(dataset)

    print(f"The length of dataset: {dataset_size}")
    print(f"Using num_voxel_bins: {cfg.dataset.num_voxel_bins}")
    print(f"Preprocessed output path: {dataset.preprocessed_path}")

    for start_idx in range(0, dataset_size, chunk_size):
        end_idx = min(start_idx + chunk_size, dataset_size)

        # Full Python command to pass to the sbatch script
        worker_command = [
            "python",
            python_script,
            "--data_root",
            data_root,
            "--start_idx",
            str(start_idx),
            "--end_idx",
            str(end_idx),
            "--config_name",
            config_name,
        ]
        if num_voxel_bins is not None:
            worker_command.extend(["--num_voxel_bins", str(num_voxel_bins)])
        worker_command.append(
            "--normalize_aux_voxel"
            if normalize_aux_voxel
            else "--no-normalize_aux_voxel"
        )
        if test:
            worker_command.extend(["--test", "--test_set_root", test_set_root])

        sbatch_command = ["sbatch", submit_script, *worker_command]

        action = "Dry run" if dry_run else "Submitting job"
        print(
            f"{action} for indices {start_idx} to {end_idx}: "
            f"{shlex.join(sbatch_command)}"
        )
        if not dry_run:
            subprocess.run(sbatch_command, check=True)
        if submit_once:
            print("submit_once enabled; stopping after the first job.")
            break


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
        default="compute_eig_values_mvsec.py",
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
        "--data_root", type=str, default="data/MVSEC", help="Root directory for data"
    )
    parser.add_argument(
        "--test_set_root",
        type=str,
        default="data/MVSEC",
        help="Root directory for test set",
    )
    parser.add_argument(
        "--num_voxel_bins",
        type=int,
        default=None,
        help="Override dataset.num_voxel_bins and pass the same override to worker jobs",
    )
    parser.add_argument(
        "--normalize_aux_voxel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize MVSEC eig/filter voxel tensors. Use --no-normalize_aux_voxel to disable.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the sbatch commands without submitting them.",
    )
    parser.add_argument(
        "--submit_once",
        action="store_true",
        help="Only submit or print the first chunk, then stop.",
    )
    args = parser.parse_args()

    # Conditional requirement check
    if args.test and not args.test_set_root:
        parser.error("--test_set_root is required when --test is set")

    print("Arguments:", args)
    
    submit_jobs(
        data_root=args.data_root,
        chunk_size=args.chunk_size,
        submit_script=args.submit_script,
        python_script=args.python_script,
        config_name=args.config_name,
        test=args.test,
        test_set_root=args.test_set_root,
        num_voxel_bins=args.num_voxel_bins,
        normalize_aux_voxel=args.normalize_aux_voxel,
        dry_run=args.dry_run,
        submit_once=args.submit_once,
    )
