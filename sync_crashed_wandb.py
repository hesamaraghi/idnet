import os
import wandb
import subprocess
import argparse


def sync_project_crashed_runs(wandb_dir, entity, project, group_id, num_groups=4):
    api = wandb.Api()
    print(f"🔍 Fetching runs from {entity}/{project}")
    runs = api.runs(f"{entity}/{project}")

    crashed_run_ids = [run.id for run in runs if run.state == "crashed"]
    print(f"Found {len(crashed_run_ids)} crashed runs on server")

    local_run_dirs = [d for d in os.listdir(wandb_dir) if d.startswith("run-")]

    selected_runs = []
    for run_id in crashed_run_ids:
        # assign group by hashing run_id
        group = hash(run_id) % num_groups
        if group != group_id:
            continue

        # find local folder for this run
        match = [d for d in local_run_dirs if d.endswith(run_id)]
        if not match:
            print(f"⚠️ No local folder found for crashed run {run_id}")
            continue
        run_path = os.path.join(wandb_dir, match[0])
        selected_runs.append((run_id, run_path))

    print(f"🗂 Group {group_id}: {len(selected_runs)} runs to sync")

    # one sbatch per run
    for run_id, run_path in selected_runs:
        sbatch_cmd = ["wandb", "sync", run_path]
        print(f"🚀 Submitting sync job for {run_id}: {' '.join(sbatch_cmd)}")
        subprocess.run(sbatch_cmd)

    print(f"✅ Submitted {len(selected_runs)} sync jobs for group {group_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_dir", type=str, default="wandb")
    parser.add_argument("--entity", type=str, required=True)
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--group_id", type=int, required=True, help="Which group to run")
    parser.add_argument("--num_groups", type=int, default=10)
    args = parser.parse_args()

    sync_project_crashed_runs(
        wandb_dir=args.wandb_dir,
        entity=args.entity,
        project=args.project,
        group_id=args.group_id,
        num_groups=args.num_groups,
    )
# Example usage:
