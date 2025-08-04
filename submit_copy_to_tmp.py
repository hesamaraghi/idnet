import os
from pathlib import Path
import subprocess
import argparse
from glob import glob


def submit_jobs(submit_script, data_root, tmp_data_root, check_preprocessed):

    data_root = Path(data_root)
    tmp_data_root = Path(tmp_data_root)
    # Check if the data_root exists, if not, use the tmp_data_root
    assert data_root.exists(), f"Data root {data_root} does not exist."
    # Check if the tmp_data_root exists, if not, create it
    # assert tmp_data_root.exists(), f"Temporary data root {tmp_data_root} does not exist."
    # Check if the submit_script exists
    assert os.path.exists(
        submit_script
    ), f"Submit script {submit_script} does not exist."
    # Check if folder "train_events" exists in data_root
    train_events_path = data_root / "train_events"
    assert (
        train_events_path.exists()
    ), f"train_events folder {train_events_path} does not exist in data root."
    # Check if folder "train_events" exists in tmp_data_root
    tmp_train_events_path = tmp_data_root / "train_events"
    # assert tmp_train_events_path.exists(), f"train_events folder {tmp_train_events_path} does not exist in temporary data root."
    # Get all sequence directories in the train_events folder that are also in the folder "train_optical_flow"
    seq_dirs = [
        f
        for f in list(train_events_path.glob("*/"))
        if (data_root / "train_optical_flow" / f.name).is_dir()
    ]
    for seq in seq_dirs:
        print(f"Found sequence directory: {seq.name}")
    # Check if each sequence directory exists in the temporary data root
    # for seq_dir in seq_dirs:
    #     seq_name = seq_dir.name
    #     tmp_seq_dir = tmp_train_events_path / seq_name
    #     assert tmp_seq_dir.exists(), f"Sequence directory {tmp_seq_dir} does not exist in temporary data root."
    # Check if each sequence directory has folder "preprocessed" in it
    for seq_dir in seq_dirs:
        seq_name = seq_dir.name
        preprocessed_path = seq_dir / "preprocessed"
        assert (
            preprocessed_path.exists()
        ), f"Preprocessed folder {preprocessed_path} does not exist in sequence directory {seq_name}."
    # For each sequence directory, submit a job to copy the preprocessed folder to the temporary data root using rsync
    for seq_dir in seq_dirs:
        seq_name = seq_dir.name
        preprocessed_path = seq_dir / "preprocessed"
        tmp_seq_dir = tmp_train_events_path / seq_name / "preprocessed"
        mkdir_command = f"mkdir -p {tmp_seq_dir}"
        # Create the command to copy the preprocessed folder to the temporary data root and if it already exists, replace it
        rsync_command = (
            f"rsync -av --no-perms --delete {preprocessed_path}/ {tmp_seq_dir}/"
        )
        rsync_command = f"{mkdir_command} && {rsync_command}"
        if check_preprocessed:
            # Check if the check_preprocessed_file.py script exists
            check_script = Path("check_preprocessed_file.py")
            assert (
                check_script.exists()
            ), f"check_preprocessed_file.py script does not exist."
            # Create the command to run the check_preprocessed_file.py script
            check_command = (
                f"python {check_script} --preprocessed_data_root {preprocessed_path}"
            )
            # Combine the rsync command and the check command
            rsync_command = f"{check_command} && {rsync_command}"
        both_command = f"{rsync_command} && echo 'Copy completed for {seq_name}'"
        
        # Full Python command to pass to the sbatch script
        sbatch_command = f"sbatch {submit_script} bash -lc \"{both_command}\""
        print(f"Submitting job for sequence {seq_name}: {sbatch_command}")
        subprocess.call(sbatch_command, shell=True)
        # break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit jobs using SLURM.")
    parser.add_argument(
        "--submit_script", type=str, required=True, help="Path to submit_job.sh"
    )
    parser.add_argument(
        "--data_root", type=str, default="data", help="Root directory for data"
    )
    parser.add_argument(
        "--tmp_data_root",
        type=str,
        default="/tmp/datasets/DSEC",
        help="Temporary data root directory",
    )
    parser.add_argument(
        "--check_preprocessed", action="store_true", help="Check preprocessed files"
    )

    args = parser.parse_args()

    submit_jobs(
        submit_script=args.submit_script,
        data_root=args.data_root,
        tmp_data_root=args.tmp_data_root,
        check_preprocessed=args.check_preprocessed,
    )
