import os
import json
import yaml
import argparse
import hashlib
import subprocess
import wandb


def get_local_runs(wandb_dir, skip_synced):
    """
    Return {run_id: run_path} for all offline runs in wandb_dir.
    """
    local_runs = {}
    for d in os.listdir(wandb_dir):
        if not d.startswith("offline-run-"):
            continue
        run_path = os.path.join(wandb_dir, d)
        if not os.path.isdir(run_path):
            continue
        if skip_synced:
            # if there is a file with extension .wandb.synced in the run directory, skip it
            if any(fname.endswith(".wandb.synced") for fname in os.listdir(run_path)):
                print(f"⏭ Skipping {d}, already synced", flush=True)
                continue
        run_id = d.split("-")[-1]
        local_runs[run_id] = run_path

    return local_runs

def get_remote_run_statuses(entity, project):
    """
    Return a dict {run_id: state} for all runs in the project on wandb server.
    """
    api = wandb.Api()
    remote_runs = {}
    try:
        for run in api.runs(f"{entity}/{project}"):
            remote_runs[run.id] = run.state  # possible states: 'running', 'crashed', 'finished'
    except wandb.errors.CommError as e:
        print(f"⚠️ Error fetching runs from W&B server: {e}", flush=True)
    return remote_runs

def sync_project_offline_runs(wandb_dir, group_id, entity, project, num_groups=4, force=False, skip_synced=False):
    local_runs = get_local_runs(wandb_dir, skip_synced)
    print(f"📂 Found {len(local_runs)} offline runs locally", flush=True)
    
    remote_runs = get_remote_run_statuses(entity, project) if not force else {}
    selected_runs = []
    
    for run_id, run_path in local_runs.items():
        group = int(hashlib.md5(run_id.encode()).hexdigest(), 16) % num_groups
        if group != group_id:
            continue

        # Skip if run exists on server and did not crash, unless force is True
        remote_state = remote_runs.get(run_id)
        if remote_state is not None and remote_state != "crashed" and not force:
            print(f"⏭ Skipping {run_id}, already on server with state: {remote_state}", flush=True)
            continue
        
        selected_runs.append((run_id, run_path))

    print(f"🗂 Group {group_id}: {len(selected_runs)} runs to sync", flush=True)

    for run_id, run_path in selected_runs:
        sbatch_cmd = ["wandb", "sync", run_path]
        print(f"🚀 Submitting sync job for {run_id}: {' '.join(sbatch_cmd)}", flush=True)
        subprocess.run(sbatch_cmd)

    print(f"✅ Submitted {len(selected_runs)} sync jobs for group {group_id}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_dir", type=str, default="wandb")
    parser.add_argument("--group_id", type=int, required=True, help="Which group to run")
    parser.add_argument("--num_groups", type=int, default=4)
    parser.add_argument("--entity", type=str, required=True, help="W&B entity name")
    parser.add_argument("--project", type=str, required=True, help="W&B project name")
    parser.add_argument("--force", action="store_true", help="Force sync all runs")
    parser.add_argument("--skip_synced", action="store_true", help="Skip runs already on server")
    args = parser.parse_args()

    wandb_dir = os.path.join(args.wandb_dir,args.project,'wandb')
    
    sync_project_offline_runs(
        wandb_dir=wandb_dir,
        group_id=args.group_id,
        entity=args.entity,
        project=args.project,
        num_groups=args.num_groups,
        force=args.force,
        skip_synced=args.skip_synced,
    )