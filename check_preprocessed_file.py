#!/usr/bin/env python3
import torch
from glob import glob
from tqdm import tqdm
import argparse


# This script checks preprocessed files for required keys and prints warnings if they are missing.
def check_preprocessed_file(file_path):
    check_failed = False
    try:
        data = torch.load(file_path)
    except Exception as e:
        print(f"FAILED to load {file_path}: {e}")
        # say that the file is not a valid .pt file
        check_failed = True
    if check_failed:
        return 1
    required_keys = [
        "eigenvalues_volume_new",
        "filter_values_volume_new",
    ]

    for key in list(data.keys()):
        if "_old" in key or "_next" in key:
            print(f"old or next in: {file_path}")
            check_failed = True
            break

    for key in required_keys:
        if key not in data:
            print(f"Missing key '{key}' in file: {file_path}")
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
            if check_preprocessed_file(file) != 0:
                files_are_valid = False
        if not files_are_valid:
            print("Some files failed the check. Please review the output above.")
            raise ValueError(
                "Preprocessed files check failed. Please review the output above."
            )
        else:
            print("All files passed the check successfully.")


# Example usage:
# python check_preprocessed_file.py --preprocessed_data_root /path/to/preprocessed/data
