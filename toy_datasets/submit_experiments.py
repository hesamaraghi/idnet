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
        "feature_type": ["both", "original"],#"both_time_augmented"],#"original_time_augmented", "original_time_augmented_repeated_augmented", "original", "filter", "eig","both"],
        "tau": [30.0],# 1.0, 5.0,
        "filter_size": [7],
        "toy_dataset": ["star8"],
        "test_train_split": ["temporal"], #"random", "temporal"
        "random_seed": [42, 420, 4200, 42000, 420000, 0, 10, 100, 1000, 10000],
        "k": [50, 100, 200],
        # Data params
        "event_generation_method": ["intensity"],  # "synthetic" or "v2e" or "intensity"
        "intensity_pos_threshold": [0.02],
        "intensity_neg_threshold": [0.02],
        # "img_size": [[256, 256]],
        # "total_frames": [2000],
        # "num_points": [5],
        # "outer_radius": [40],
        # "inner_radius": [20],
        # "num_rotations": [2],
        
        # Texture params (optional - uncomment to use)
        # "use_random_dtd_texture": [True],
        # "dtd_texture_mode": ["foreground"],  # "foreground", "background", or "both"
        "foreground_texture": ["image"],
        "fg_image_path": ["data/dtd/images/woven/woven_0053.jpg"],
        
        "create_animation": [True],
        
        # V2E params
        # "v2e_pos_thres": [0.2],
        # "v2e_neg_thres": [0.2],
        # "v2e_fg_gamma": [2.0],
        # "v2e_bg_gamma": [0.6],
        # "v2e_temporal_filter_percent": [2.0, 4.0],  # Keep first X% of each frame
        
        # Dataset params
        "test_train_split": ["temporal"],
        "test_size": [0.2],
        
        # Model params
        "hidden_dim": [128],
        "lr": [1e-4],
        "max_epochs": [500],
        "batch_size": [1024], 
        "relative_coordinates": [True],
        "project": ["knn-mlp-regression-relative-multiseed-texture-intensity"],
        "online" : [True],
    }
    # search_space = {
    #     "entity": ["haraghi"],
    #     "project": ["knn-mlp-regression-relative-multiseed"],
    #     "eval_run_id": [
    #         "7gp7i7wa",
    #         # "jq88h8jp",
    #         # "top9wmcg",
    #     ]
    # }

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
