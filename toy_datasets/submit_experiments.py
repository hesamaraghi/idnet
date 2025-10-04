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
    # search_space = {
    #     "feature_type": ["original", "filter", "eig", "both"],
    #     "tau": [30.0],# 1.0, 5.0,
    #     "filter_size": [7],
    #     "toy_dataset": ["star8"],
    #     "test_train_split": ["temporal", "random"],
    #     "test_split_seed": [42, 420, 4200, 42000, 420000, 0, 10, 100, 1000, 10000],
    #     "k": [50],
    #     "hidden_dim": [128],
    #     "lr": [1e-4],
    #     "max_epochs": [200],
    #     "batch_size": [16], 
    #     "relative_coordinates": [True],
    #     "project": ["knn-mlp-regression-relative-multiseed"],
    #     "online" : [False],
    # }
    search_space = {
        "entity": ["haraghi"],
        "project": ["knn-mlp-regression-relative-multiseed"],
        "eval_run_id": [
            "6n50ipg0",
            "l4vl3tum",
            "ida8pz13",
            "6nuu97pl",
            "aogolt51",
            "etc9hhdr",
            "mas120q7",
            "f3iylfdt",
            "uwlqh0qj",
            "30ygqqp2",
            "irkgq5hv",
            "ejaj3r80",
            "495wwpmm",
            "v99oouip",
            "ecv2yq14",
            "a2ljsc1c",
            "zikvh1fa",
            "bg506vjw",
            "lxidc3zr",
            "no30gesr",
            "kfo7q4e3",
            "m0dh8ica",
        ]
    }

    # -----------------------------------------
    # Generate all combinations
    # -----------------------------------------
    keys = list(search_space.keys())
    values = list(search_space.values())
    all_combinations = list(itertools.product(*values))
    # Shuffle combinations to mix different hyperparameters
    import random
    random.shuffle(all_combinations)

    print(f"🔍 Launching {len(all_combinations)} jobs")

    for combo in all_combinations:
        exp_dict = dict(zip(keys, combo))

        # build python command
        exp_args = []
        for k, v in exp_dict.items():
            if isinstance(v, bool):
                if v:
                    exp_args.append(f"--{k}")
            else:
                exp_args.append(f"--{k} {v}")

        cmd = f"python {args.train_script} {' '.join(exp_args)}"

        # sbatch command
        sbatch_cmd = f"sbatch {args.sbatch_script} {cmd}"

        print(f"🚀 Submitting: {sbatch_cmd}")
        
        if args.dry_run:
            continue
        
        subprocess.call(sbatch_cmd, shell=True) 

if __name__ == "__main__":
    main()
