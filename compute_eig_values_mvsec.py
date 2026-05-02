import os
from omegaconf import OmegaConf
from tqdm import tqdm
from pathlib import Path
from hydra import initialize_config_dir, compose
import argparse

from idn.loader.loader_mvsec import MVSEC


def configure_mvsec_dataset(cfg, data_root, test_set_root, num_voxel_bins=None):
    cfg.dataset.force_preprocess = True
    cfg.dataset.in_memory = False
    cfg.dataset.do_not_save_preprocessed = False
    cfg.dataset.add_eigenvalues = True
    cfg.dataset.add_filter_values = True
    cfg.dataset.common.data_root = data_root
    cfg.dataset.common.test_root = test_set_root
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

# Parse command-line arguments for HPC jobs
parser = argparse.ArgumentParser(description="Process dataset indices for HPC")
parser.add_argument(
    "--start_idx", type=int, default=0, help="Start index of dataset range"
)
parser.add_argument("--end_idx", type=int, default=1, help="End index of dataset range")
parser.add_argument(
    "--config_name", type=str, required=True, help="Name of the config file"
)
parser.add_argument("--data_root", type=str, default="data/MVSEC", help="Root directory for data")
parser.add_argument("--test", action="store_true", help="Run in test mode with a small dataset")
parser.add_argument("--test_set_root", type=str, default="data/MVSEC", help="Root directory for test set")
parser.add_argument("--num_voxel_bins", type=int, default=None, help="Override dataset.num_voxel_bins")
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

configure_mvsec_dataset(
    cfg=cfg,
    data_root=args.data_root,
    test_set_root=args.test_set_root,
    num_voxel_bins=args.num_voxel_bins,
)

print(OmegaConf.to_yaml(cfg))

if args.test:
    print(f"Using test set root: {args.test_set_root}")
    if "mvsec" not in cfg.validation:
        raise KeyError(
            f"Config '{args.config_name}' does not define validation.mvsec. "
            "Use mvsec_train_all or update the validation package name."
        )
    cfg.validation.mvsec.dataset.common.data_root = args.test_set_root
    cfg.validation.mvsec.dataset.common.test_root = args.test_set_root
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


datasets_len = len(dataset)

print(f"The length of dataset: {datasets_len}")
print(f"Using num_voxel_bins: {cfg.dataset.num_voxel_bins}")
print(f"Overwrite enabled: force_preprocess={cfg.dataset.force_preprocess}, do_not_save_preprocessed={cfg.dataset.do_not_save_preprocessed}")
print(f"Preprocessed output path: {dataset.preprocessed_path}")

start_idx = args.start_idx
end_idx = args.end_idx

print(f"Processing dataset indices from {start_idx} t/m {end_idx - 1}")

assert 0 <= start_idx < datasets_len, "Start index out of bounds"
assert 0 < end_idx <= datasets_len, "End index out of bounds"
assert start_idx < end_idx, "Start index must be less than end index"

assert cfg.dataset.get("add_eigenvalues", False) or cfg.dataset.get("add_filter_values", False), \
    "You need to set add_eigenvalues or add_filter_values to True in your config file."
add_eigenvalues = cfg.dataset.get("add_eigenvalues", False)
add_filter_values = cfg.dataset.get("add_filter_values", False)
print(f"add_eigenvalues: {add_eigenvalues}, add_filter_values: {add_filter_values}")

# Process each item in the specified range
for i in tqdm(range(start_idx, end_idx), desc="Processing samples"):
    try:
        sample = dataset[i]
        validate_sample_shapes(sample, cfg.dataset.num_voxel_bins, i)
        # seq_name = cfg.dataset.val.seq if args.test else cfg.dataset.train.seq
        seq_name = dataset.seq_name
        # Do your processing here
        if i % 10 == 0:
            matplotlib_cache = Path(os.environ.get("MPLCONFIGDIR", "cache/matplotlib"))
            matplotlib_cache.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            seq_path = Path(dataset.seq_path) / seq_name
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
                
            for channel in visualization_channels(cfg.dataset.num_voxel_bins):
                file_path = (
                    event_voxel_path
                    / f"event_voxel_{i}_{channel}.png"
                )
                fig, ax = plt.subplots(1, 1, figsize=(10, 7))
                im = ax.imshow(sample["event_volume_new"][channel, :, :], cmap="binary")
                ax.set_title(f"Augmented event voxel channel {channel}")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                plt.tight_layout()
                plt.savefig(file_path)
                plt.close()
                if add_eigenvalues:
                    file_path = eig_1_path / f"eig_1_{i}_{channel}.png"
                    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
                    im = ax.imshow(
                        sample["eigenvalues_volume_new"][channel, :, :], cmap="binary"
                    )
                    ax.set_title(f"Augmented eig value 1 channel {channel}")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    plt.tight_layout()
                    plt.savefig(file_path)
                    plt.close()

                    file_path = eig_2_path / f"eig_2_{i}_{channel}.png"
                    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
                    im = ax.imshow(
                        sample["eigenvalues_volume_new"][channel + cfg.dataset.num_voxel_bins, :, :], cmap="binary"
                    )
                    ax.set_title(f"Augmented eig value 2 channel {channel}")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    plt.tight_layout()
                    plt.savefig(file_path)
                    plt.close()
                if add_filter_values:
                    file_path = filter_value_path / f"filter_values_{i}_{channel}.png"
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
        raise RuntimeError(f"MVSEC preprocessing failed for index {i}") from e
