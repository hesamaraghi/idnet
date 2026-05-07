import argparse
import os
import shlex
import subprocess
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from idn.loader.loader_evimo import assemble_evimo_sequences


def configure_evimo_dataset(
    cfg,
    data_root,
    split,
    sequences,
    preprocessed_root=None,
    num_voxel_bins=None,
    normalize_aux_voxel=True,
):
    cfg.dataset.force_preprocess = True
    cfg.dataset.in_memory = False
    cfg.dataset.do_not_save_preprocessed = False
    cfg.dataset.add_eigenvalues = True
    cfg.dataset.add_filter_values = True
    cfg.dataset.normalize_aux_voxel = normalize_aux_voxel
    cfg.dataset.common.data_root = data_root
    cfg.dataset.common.test_root = str(Path(data_root) / split)
    cfg.dataset.split = split
    cfg.dataset.val.split = split
    if sequences is not None:
        cfg.dataset.val.seq = sequences
    if preprocessed_root is not None:
        cfg.dataset.common.preprocessed_root = preprocessed_root
    if num_voxel_bins is not None:
        cfg.dataset.num_voxel_bins = num_voxel_bins


def parse_sequences(values):
    if not values:
        return None
    if len(values) == 1 and "," in values[0]:
        return [value for value in values[0].split(",") if value]
    return values


def pixi_python_command():
    return [
        "pixi",
        "run",
        "--manifest-path",
        "__ignore_blackwell_pixi/pixi.toml",
        "python",
    ]


def submit_jobs(
    data_root,
    split,
    sequences,
    chunk_size,
    submit_script,
    python_script,
    config_name,
    preprocessed_root=None,
    visualization_root=None,
    num_voxel_bins=None,
    normalize_aux_voxel=True,
    visualize_every=10,
    no_visualize=False,
    dry_run=False,
    submit_once=False,
):
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if not Path(submit_script).is_file():
        raise FileNotFoundError(f"Submit script does not exist: {submit_script}")
    if not Path(python_script).is_file():
        raise FileNotFoundError(f"Python worker script does not exist: {python_script}")

    config_dir = os.path.abspath("idn/config")
    print("Loading configs from:", config_dir)
    with initialize_config_dir(
        config_dir=config_dir, job_name="evimo_submit_eig_values", version_base=None
    ):
        cfg = compose(config_name=config_name)

    configure_evimo_dataset(
        cfg=cfg,
        data_root=data_root,
        split=split,
        sequences=sequences,
        preprocessed_root=preprocessed_root,
        num_voxel_bins=num_voxel_bins,
        normalize_aux_voxel=normalize_aux_voxel,
    )
    print(OmegaConf.to_yaml(cfg))

    datasets = assemble_evimo_sequences(
        cfg.dataset.common.data_root,
        split=cfg.dataset.val.split,
        include_seq=cfg.dataset.val.seq,
        config=cfg.dataset,
        num_bins=cfg.dataset.get("num_voxel_bins", None),
    )
    dataset_size = sum(len(dataset) for dataset in datasets)

    print(f"The length of dataset: {dataset_size}")
    print(f"Using num_voxel_bins: {cfg.dataset.num_voxel_bins}")
    for dataset in datasets:
        print(f"{dataset.seq_name}: samples={len(dataset)}, preprocessed={dataset.preprocessed_path}")

    for start_idx in range(0, dataset_size, chunk_size):
        end_idx = min(start_idx + chunk_size, dataset_size)
        worker_command = [
            *pixi_python_command(),
            python_script,
            "--data_root",
            data_root,
            "--split",
            split,
            "--start_idx",
            str(start_idx),
            "--end_idx",
            str(end_idx),
            "--config_name",
            config_name,
            "--visualize_every",
            str(visualize_every),
        ]
        if sequences:
            worker_command.append("--seq")
            worker_command.extend(sequences)
        if preprocessed_root is not None:
            worker_command.extend(["--preprocessed_root", preprocessed_root])
        if visualization_root is not None:
            worker_command.extend(["--visualization_root", visualization_root])
        if num_voxel_bins is not None:
            worker_command.extend(["--num_voxel_bins", str(num_voxel_bins)])
        worker_command.append(
            "--normalize_aux_voxel"
            if normalize_aux_voxel
            else "--no-normalize_aux_voxel"
        )
        if no_visualize:
            worker_command.append("--no_visualize")

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
    parser = argparse.ArgumentParser(description="Submit EVIMO2v2 eig/filter jobs using SLURM.")
    parser.add_argument(
        "--chunk_size",
        "--chunk-size",
        dest="chunk_size",
        type=int,
        required=True,
        help="Number of samples per job",
    )
    parser.add_argument(
        "--submit_script",
        "--submit-script",
        dest="submit_script",
        type=str,
        required=True,
        help="Path to submit_job.sh",
    )
    parser.add_argument(
        "--python_script",
        "--python-script",
        dest="python_script",
        type=str,
        default="compute_eig_values_evimo.py",
        help="Path to the processing Python script",
    )
    parser.add_argument(
        "--config_name",
        "--config-name",
        dest="config_name",
        type=str,
        default="id_eval_evimo",
    )
    parser.add_argument(
        "--data_root",
        "--data-root",
        dest="data_root",
        type=str,
        default="data/EVIMO2v2/samsung_mono/imo",
        help="EVIMO2v2 root containing train/ and eval/ splits",
    )
    parser.add_argument("--split", type=str, default="eval", choices=("train", "eval"))
    parser.add_argument("--seq", nargs="*", default=None, help="Optional sequence names")
    parser.add_argument(
        "--preprocessed_root",
        "--preprocessed-root",
        dest="preprocessed_root",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--visualization_root",
        "--visualization-root",
        dest="visualization_root",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--num_voxel_bins",
        "--num-voxel-bins",
        dest="num_voxel_bins",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--normalize_aux_voxel",
        "--normalize-aux-voxel",
        dest="normalize_aux_voxel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize EVIMO2v2 eig/filter voxel tensors. Use --no-normalize_aux_voxel to disable.",
    )
    parser.add_argument(
        "--visualize_every",
        "--visualize-every",
        dest="visualize_every",
        type=int,
        default=10,
    )
    parser.add_argument("--no_visualize", "--no-visualize", dest="no_visualize", action="store_true")
    parser.add_argument("--dry_run", "--dry-run", dest="dry_run", action="store_true")
    parser.add_argument("--submit_once", "--submit-once", dest="submit_once", action="store_true")
    args = parser.parse_args()

    print("Arguments:", args)
    submit_jobs(
        data_root=args.data_root,
        split=args.split,
        sequences=parse_sequences(args.seq),
        chunk_size=args.chunk_size,
        submit_script=args.submit_script,
        python_script=args.python_script,
        config_name=args.config_name,
        preprocessed_root=args.preprocessed_root,
        visualization_root=args.visualization_root,
        num_voxel_bins=args.num_voxel_bins,
        normalize_aux_voxel=args.normalize_aux_voxel,
        visualize_every=args.visualize_every,
        no_visualize=args.no_visualize,
        dry_run=args.dry_run,
        submit_once=args.submit_once,
    )
