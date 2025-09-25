import os
import subprocess
import wandb


def sync_project_crashed_runs(wandb_dir="wandb", entity=None, project=None):
    api = wandb.Api()
    print(f"🔍 Fetching runs from {entity}/{project}")
    runs = api.runs(f"{entity}/{project}")

    crashed_run_ids = [run.id for run in runs if run.state == "crashed"]
    print(f"Found {len(crashed_run_ids)} crashed runs on server")

    local_run_dirs = [d for d in os.listdir(wandb_dir) if d.startswith("run-")]

    synced = 0
    for run_id in crashed_run_ids:
        # match run id with local folder suffix
        match = [d for d in local_run_dirs if d.endswith(run_id)]
        if not match:
            print(f"⚠️ No local folder found for crashed run {run_id}")
            continue

        run_path = os.path.join(wandb_dir, match[0])
        print(f"🔄 Syncing {run_path} -> {entity}/{project}/{run_id}")
        subprocess.run(["wandb", "sync", run_path])
        synced += 1

    print(f"✅ Synced {synced}/{len(crashed_run_ids)} crashed runs")


if __name__ == "__main__":
    # ⚠️ Fill these in with your project info
    ENTITY = "haraghi"
    PROJECT = "knn-mlp-regression"

    sync_project_crashed_runs(wandb_dir="wandb", entity=ENTITY, project=PROJECT)

