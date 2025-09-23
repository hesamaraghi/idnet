import os
from pathlib import Path
import argparse


def generate_commands(data_root, tmp_data_root, test=False, check_preprocessed=False, output_file="copy_commands.sh"):
    if test:
        subfolder = "test"
    else:
        subfolder = "train_events"

    data_root = Path(data_root)
    tmp_data_root = Path(tmp_data_root)

    events_path = data_root / subfolder
    assert events_path.exists(), f"Folder '{events_path}' does not exist."

    # Collect sequence dirs
    seq_dirs = [
        f for f in list(events_path.glob("*/"))
        if (data_root / "train_optical_flow" / f.name).is_dir() or test
    ]

    with open(output_file, "w") as f:
        f.write("#!/bin/bash\n\n")
        for seq_dir in seq_dirs:
            seq_name = seq_dir.name
            preprocessed_path = seq_dir / "preprocessed"
            assert preprocessed_path.exists(), f"No preprocessed folder for sequence '{seq_name}': {preprocessed_path}"

            tmp_seq_dir = tmp_data_root / subfolder / seq_name / "preprocessed"
            mkdir_cmd = f"mkdir -p {tmp_seq_dir}"
            rsync_cmd = f"rsync -avz --no-perms {preprocessed_path}/ {tmp_seq_dir}/"
            full_cmd = f"{mkdir_cmd} && {rsync_cmd} && echo 'Copy completed for {seq_name}'"

            f.write(full_cmd + "\n")

    print(f"✅ Wrote rsync commands to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--tmp_data_root", type=str, default="/tmp/maraghi/datasets/DSEC")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--check_preprocessed", action="store_true")
    parser.add_argument("--output_file", type=str, default="copy_commands.sh")
    args = parser.parse_args()

    generate_commands(
        data_root=args.data_root,
        tmp_data_root=args.tmp_data_root,
        test=args.test,
        check_preprocessed=args.check_preprocessed,
        output_file=args.output_file,
    )
