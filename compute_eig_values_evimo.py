import argparse
import os
from bisect import bisect_right
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tqdm import tqdm

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


def validate_sample_shapes(sample, num_voxel_bins, sample_idx):
    expected_shapes = {
        "event_volume_new": num_voxel_bins,
        "eigenvalues_volume_new": 2 * num_voxel_bins,
        "filter_values_volume_new": num_voxel_bins,
    }
    for key, expected_channels in expected_shapes.items():
        if key not in sample:
            raise KeyError(f"Missing '{key}' for sample {sample_idx}")
        if sample[key].shape[0] != expected_channels:
            raise ValueError(
                f"Sample {sample_idx} has {key} shape {tuple(sample[key].shape)}, "
                f"expected {expected_channels} channels for num_voxel_bins={num_voxel_bins}"
            )


def visualization_channels(num_voxel_bins):
    step = max(1, num_voxel_bins // 3)
    return range(0, num_voxel_bins, step)


def cumulative_lengths(datasets):
    lengths = []
    total = 0
    for dataset in datasets:
        total += len(dataset)
        lengths.append(total)
    return lengths


def resolve_dataset_index(datasets, cumulative, global_idx):
    dataset_idx = bisect_right(cumulative, global_idx)
    previous = 0 if dataset_idx == 0 else cumulative[dataset_idx - 1]
    return datasets[dataset_idx], global_idx - previous


def save_visualizations(sample, output_root, split, num_voxel_bins, global_idx):
    matplotlib_cache = Path(os.environ.get("MPLCONFIGDIR", "cache/matplotlib"))
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seq_path = Path(output_root) / split / sample["seq_name"]
    event_voxel_path = seq_path / "event_voxel"
    eig_1_path = seq_path / "eig_1"
    eig_2_path = seq_path / "eig_2"
    filter_value_path = seq_path / "filter_values"
    for path in (event_voxel_path, eig_1_path, eig_2_path, filter_value_path):
        path.mkdir(parents=True, exist_ok=True)

    file_idx = sample["file_index"]
    for channel in visualization_channels(num_voxel_bins):
        file_path = event_voxel_path / f"event_voxel_{global_idx}_{file_idx}_{channel}.png"
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        im = ax.imshow(sample["event_volume_new"][channel, :, :], cmap="binary")
        ax.set_title(f"EVIMOv2 event voxel channel {channel}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()

        file_path = eig_1_path / f"eig_1_{global_idx}_{file_idx}_{channel}.png"
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        im = ax.imshow(sample["eigenvalues_volume_new"][channel, :, :], cmap="binary")
        ax.set_title(f"EVIMOv2 eig value 1 channel {channel}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()

        file_path = eig_2_path / f"eig_2_{global_idx}_{file_idx}_{channel}.png"
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        im = ax.imshow(
            sample["eigenvalues_volume_new"][channel + num_voxel_bins, :, :],
            cmap="binary",
        )
        ax.set_title(f"EVIMOv2 eig value 2 channel {channel}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()

        file_path = filter_value_path / f"filter_values_{global_idx}_{file_idx}_{channel}.png"
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        im = ax.imshow(sample["filter_values_volume_new"][channel, :, :], cmap="binary")
        ax.set_title(f"EVIMOv2 filter values channel {channel}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()


def parse_sequences(values):
    if not values:
        return None
    if len(values) == 1 and "," in values[0]:
        return [value for value in values[0].split(",") if value]
    return values


def main():
    parser = argparse.ArgumentParser(description="Precompute EVIMOv2 eig/filter values.")
    parser.add_argument(
        "--start_idx",
        "--start-idx",
        dest="start_idx",
        type=int,
        default=0,
        help="Start global dataset index",
    )
    parser.add_argument(
        "--end_idx",
        "--end-idx",
        dest="end_idx",
        type=int,
        default=1,
        help="Exclusive end global dataset index",
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
        default="data/EVIMOv2/samsung_mono/imo",
        help="EVIMOv2 root containing train/ and eval/ splits",
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
        help="Normalize EVIMOv2 eig/filter voxel tensors. Use --no-normalize_aux_voxel to disable.",
    )
    parser.add_argument(
        "--visualize_every",
        "--visualize-every",
        dest="visualize_every",
        type=int,
        default=10,
    )
    parser.add_argument("--no_visualize", "--no-visualize", dest="no_visualize", action="store_true")
    args = parser.parse_args()

    config_dir = os.path.abspath("idn/config")
    print("Loading configs from:", config_dir)
    with initialize_config_dir(
        config_dir=config_dir, job_name="evimo_eig_values", version_base=None
    ):
        cfg = compose(config_name=args.config_name)

    sequences = parse_sequences(args.seq)
    configure_evimo_dataset(
        cfg=cfg,
        data_root=args.data_root,
        split=args.split,
        sequences=sequences,
        preprocessed_root=args.preprocessed_root,
        num_voxel_bins=args.num_voxel_bins,
        normalize_aux_voxel=args.normalize_aux_voxel,
    )
    print(OmegaConf.to_yaml(cfg))

    datasets = assemble_evimo_sequences(
        cfg.dataset.common.data_root,
        split=cfg.dataset.val.split,
        include_seq=cfg.dataset.val.seq,
        config=cfg.dataset,
        num_bins=cfg.dataset.get("num_voxel_bins", None),
    )
    cumulative = cumulative_lengths(datasets)
    dataset_size = cumulative[-1]

    print(f"The length of dataset: {dataset_size}")
    print(f"Using num_voxel_bins: {cfg.dataset.num_voxel_bins}")
    print(f"normalize_aux_voxel: {cfg.dataset.normalize_aux_voxel}")
    print(
        "Overwrite enabled: "
        f"force_preprocess={cfg.dataset.force_preprocess}, "
        f"do_not_save_preprocessed={cfg.dataset.do_not_save_preprocessed}"
    )
    for dataset in datasets:
        print(f"{dataset.seq_name}: samples={len(dataset)}, preprocessed={dataset.preprocessed_path}")

    start_idx = args.start_idx
    end_idx = args.end_idx
    print(f"Processing dataset indices from {start_idx} t/m {end_idx - 1}")

    assert 0 <= start_idx < dataset_size, "Start index out of bounds"
    assert 0 < end_idx <= dataset_size, "End index out of bounds"
    assert start_idx < end_idx, "Start index must be less than end index"
    assert cfg.dataset.get("add_eigenvalues", False) or cfg.dataset.get(
        "add_filter_values", False
    ), "You need to set add_eigenvalues or add_filter_values to True."

    visualization_root = args.visualization_root or cfg.dataset.common.data_root
    for global_idx in tqdm(range(start_idx, end_idx), desc="Processing samples"):
        try:
            dataset, local_idx = resolve_dataset_index(datasets, cumulative, global_idx)
            sample = dataset[local_idx]
            validate_sample_shapes(sample, cfg.dataset.num_voxel_bins, global_idx)
            if (
                not args.no_visualize
                and args.visualize_every > 0
                and global_idx % args.visualize_every == 0
            ):
                save_visualizations(
                    sample=sample,
                    output_root=visualization_root,
                    split=args.split,
                    num_voxel_bins=cfg.dataset.num_voxel_bins,
                    global_idx=global_idx,
                )
            print(
                f"Index {global_idx} processed "
                f"({sample['seq_name']}, local={local_idx}, file={sample['file_index']})."
            )
        except Exception as exc:
            print(f"Error processing index {global_idx}: {exc}")
            raise RuntimeError(f"EVIMOv2 preprocessing failed for index {global_idx}") from exc


if __name__ == "__main__":
    main()
