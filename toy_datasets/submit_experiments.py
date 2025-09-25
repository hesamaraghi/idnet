import itertools
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sbatch_script", type=str, default="sbatch_folder/small_cpu.sbatch",
                        help="Path to sbatch script")
    parser.add_argument("--train_script", type=str, default="toy_datasets/knn_mlp.py",
                        help="Training python script")
    parser.add_argument("--dry_run", action="store_true",
                        help="If set, only print commands without executing")
    args = parser.parse_args()

    # -----------------------------------------
    # Define search space (edit these lists)
    # -----------------------------------------
    search_space = {
        "feature_type": ["original", "eig", "filter", "both"],
        "tau": [1.0, 5.0, 30.0],
        "filter_size": [5, 7],
        "toy_dataset": ["star8"],
        "test_train_split": ["random", "temporal"],
        "test_split_seed": [42],
        "k": [1, 3, 5, 10, 20],
        "hidden_dim": [64, 128],
        "lr": [1e-3, 1e-4],
        "max_epochs": [200],
        "batch_size": [16, 32], 
    }

    # -----------------------------------------
    # Generate all combinations
    # -----------------------------------------
    keys = list(search_space.keys())
    values = list(search_space.values())
    all_combinations = list(itertools.product(*values))

    print(f"🔍 Launching {len(all_combinations)} jobs")

    for combo in all_combinations:
        exp_dict = dict(zip(keys, combo))

        # build python command
        exp_args = " ".join([f"--{k} {v}" for k, v in exp_dict.items()])
        cmd = f"python {args.train_script} {exp_args}"

        # sbatch command
        sbatch_cmd = f"sbatch {args.sbatch_script} {cmd}"

        print(f"🚀 Submitting: {sbatch_cmd}")
        
        if args.dry_run:
            continue
        subprocess.call(sbatch_cmd, shell=True)

if __name__ == "__main__":
    main()
