#!/usr/bin/env python3
import torch
from glob import glob
from tqdm import tqdm
import argparse


# This script checks preprocessed files for required keys and prints warnings if they are missing.
def _check_tensor_channels(data, file_path, key, expected_channels):
    if key not in data:
        print(f"Missing key '{key}' in file: {file_path}")
        return 1
    value = data[key]
    if not torch.is_tensor(value):
        print(f"Key '{key}' is not a tensor in file: {file_path}")
        return 1
    if value.ndim < 3:
        print(f"Key '{key}' has invalid shape {tuple(value.shape)} in file: {file_path}")
        return 1
    if expected_channels is not None and value.shape[0] != expected_channels:
        print(
            f"Key '{key}' has shape {tuple(value.shape)} in file: {file_path}; "
            f"expected {expected_channels} channels"
        )
        return 1
    return 0


def check_preprocessed_file(file_path, num_voxel_bins=None):
    check_failed = False
    try:
        data = torch.load(file_path, weights_only=False)
    except Exception as e:
        print(f"FAILED to load {file_path}: {e}")
        # say that the file is not a valid .pt file
        check_failed = True
    if check_failed:
        return 1

    for key in list(data.keys()):
        if "_old" in key or "_next" in key:
            print(f"old or next in: {file_path}")
            check_failed = True
            break

    expected_channels = {
        "event_volume_new": num_voxel_bins,
        "eigenvalues_volume_new": 2 * num_voxel_bins if num_voxel_bins is not None else None,
        "filter_values_volume_new": num_voxel_bins,
    }
    for key, channels in expected_channels.items():
        if _check_tensor_channels(data, file_path, key, channels) != 0:
            check_failed = True

    flow_key = "flow_gt_event_volume_new"
    if flow_key not in data:
        print(f"Missing key '{flow_key}' in file: {file_path}")
        check_failed = True
    elif not isinstance(data[flow_key], (tuple, list)) or len(data[flow_key]) != 2:
        print(f"Key '{flow_key}' is not a flow/mask pair in file: {file_path}")
        check_failed = True
    else:
        flow, mask = data[flow_key]
        if not torch.is_tensor(flow) or flow.ndim != 3 or flow.shape[0] != 2:
            print(f"Key '{flow_key}' has invalid flow tensor in file: {file_path}")
            check_failed = True
        if not torch.is_tensor(mask) or mask.ndim != 3 or mask.shape[0] != 1:
            print(f"Key '{flow_key}' has invalid mask tensor in file: {file_path}")
            check_failed = True

    if check_failed:
        print(f"Check failed for file: {file_path}")
        return 1
    else:
        print(f"Check passed for file: {file_path}")
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check preprocessed files for required keys."
    )
    parser.add_argument(
        "--preprocessed_data_root",
        required=True,
        type=str,
        help="Root directory for preprocessed data files.",
    )
    parser.add_argument(
        "--num_voxel_bins",
        type=int,
        default=None,
        help="Expected number of event voxel bins; use 15 for the MVSEC rerun.",
    )
    args = parser.parse_args()
    preprocessed_data_root = args.preprocessed_data_root
    all_pt_files = glob(f"{preprocessed_data_root}/*.pt")
    if not all_pt_files:
        raise ValueError(
            f"No .pt files found in {preprocessed_data_root}. Please check the directory path."
        )
    else:
        print(f"Found {len(all_pt_files)} .pt files in {preprocessed_data_root}.")
        print("Checking each file for required keys...")
        files_are_valid = True
        for file in tqdm(all_pt_files):
            if check_preprocessed_file(file, num_voxel_bins=args.num_voxel_bins) != 0:
                files_are_valid = False
        if not files_are_valid:
            print("Some files failed the check. Please review the output above.")
            raise ValueError(
                "Preprocessed files check failed. Please review the output above."
            )
        else:
            print("All files passed the check successfully.")


# Example usage:
# python check_preprocessed_file.py --preprocessed_data_root /path/to/preprocessed/data --num_voxel_bins 15
