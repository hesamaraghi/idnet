import argparse
import os

from omegaconf import OmegaConf

from idn.loader.loader_evimo import assemble_evimo_sequences
from idn.loader.loader_dsec import train_collate


def tensor_summary(name, value):
    shape = tuple(value.shape)
    if value.numel() == 0:
        return f"{name}: shape={shape}, empty"
    return (
        f"{name}: shape={shape}, dtype={value.dtype}, "
        f"min={value.min().item():.6g}, max={value.max().item():.6g}"
    )


def main():
    parser = argparse.ArgumentParser(description="Smoke test the EVIMO2v2 loader.")
    parser.add_argument(
        "--data-root",
        default="data/EVIMO2v2/samsung_mono/imo",
        help="EVIMO2v2 root containing train/ and eval/ splits.",
    )
    parser.add_argument("--split", default="eval")
    parser.add_argument("--seq", default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-voxel-bins", type=int, default=15)
    parser.add_argument("--add-eigenvalues", action="store_true")
    parser.add_argument("--add-filter-values", action="store_true")
    parser.add_argument("--tau", type=float, default=5000)
    parser.add_argument("--filter-size", type=int, default=5)
    parser.add_argument(
        "--preprocessed-root",
        default=os.environ.get("EVIMO2V2_PREPROCESSED_ROOT", "data/EVIMO2v2/preprocessed"),
        help="EVIMO2v2 preprocessed root containing train/ and eval/ splits.",
    )
    parser.add_argument(
        "--use-preprocessed-cache",
        action="store_true",
        help="Load cached .pt samples from --preprocessed-root when available.",
    )
    args = parser.parse_args()

    config = OmegaConf.create(
        {
            "dataset_name": "evimo2v2",
            "num_voxel_bins": args.num_voxel_bins,
            "image_height": 480,
            "image_width": 640,
            "add_eigenvalues": args.add_eigenvalues,
            "add_filter_values": args.add_filter_values,
            "tau": args.tau,
            "filter_size": args.filter_size,
            "in_memory": False,
            "force_preprocess": False,
            "do_not_save_preprocessed": not args.use_preprocessed_cache,
            "normalize_voxel": True,
            "normalize_aux_voxel": True,
            "skip_invalid": True,
            "common": {
                "data_root": args.data_root,
                "preprocessed_root": args.preprocessed_root,
            },
        }
    )

    datasets = assemble_evimo_sequences(
        args.data_root,
        split=args.split,
        include_seq=args.seq,
        config=config,
        num_bins=args.num_voxel_bins,
    )
    dataset = datasets[0]
    idx = min(args.sample_index, len(dataset) - 1)
    sample = dataset[idx]
    batch = train_collate([sample])

    print(f"sequence={dataset.seq_name}")
    print(f"samples={len(dataset)}")
    print(f"sample_index={idx}")
    print(f"file_index={sample['file_index']}")
    print(tensor_summary("event_volume_new", sample["event_volume_new"]))

    flow, mask = sample["flow_gt_event_volume_new"]
    print(tensor_summary("flow_gt_event_volume_new", flow))
    print(
        "flow_gt_event_volume_new_valid_mask: "
        f"shape={tuple(mask.shape)}, valid_pixels={int(mask.sum().item())}"
    )

    if args.add_eigenvalues:
        print(tensor_summary("eigenvalues_volume_new", sample["eigenvalues_volume_new"]))
    if args.add_filter_values:
        print(tensor_summary("filter_values_volume_new", sample["filter_values_volume_new"]))

    print(f"collated_event_volume_new={tuple(batch['event_volume_new'].shape)}")
    print(f"collated_flow_gt={tuple(batch['flow_gt_event_volume_new'].shape)}")
    print(
        "collated_flow_gt_mask="
        f"{tuple(batch['flow_gt_event_volume_new_valid_mask'].shape)}"
    )


if __name__ == "__main__":
    main()
