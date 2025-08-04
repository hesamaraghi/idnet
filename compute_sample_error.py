# main_script.py

import argparse
import os
import numpy as np
import torch
import json

import wandb
from omegaconf import OmegaConf

from idn.model.loss import compute_npe, sparse_lnorm
from idn.utils.trainer import Trainer
from idn.utils.helper_functions import move_batch_to_cuda
from idn.loader.loader_dsec import (
    Sequence,
    RepresentationType,
    DatasetProvider,
    assemble_dsec_sequences,
    assemble_dsec_test_set,
    train_collate,
    rec_train_collate
)
   
def compute_errors(flow_pred, flow_gt, valid_mask, save_path=None, seq_name_idx=None):
    L1_error_results = sparse_lnorm(1, flow_pred, flow_gt, valid_mask, per_frame=True)
    emap = L1_error_results["t_emap"]
    L1_error = L1_error_results["metric"]
    L2_error_results = sparse_lnorm(2, flow_pred, flow_gt, valid_mask, per_frame=True)
    L2_error = L2_error_results["metric"]
    npe_1_error = compute_npe(1, flow_pred, flow_gt, valid_mask)
    npe_3_error = compute_npe(3, flow_pred, flow_gt, valid_mask)
    
    emap_path = os.path.join(save_path, "emap")
    if emap_path is not None:
        # Save emap as a numpy file
        if seq_name_idx is not None:
            seq_name, idx = seq_name_idx
            os.makedirs(emap_path, exist_ok=True)
            np.save(os.path.join(emap_path, f"{seq_name}_{idx}_emap.npy"), emap.clone().detach().cpu().numpy())
    save_path_flow_unmasked = os.path.join(save_path, "flow_unmasked")
    if save_path_flow_unmasked is not None:
        # Save flow prediction as a numpy file
        if seq_name_idx is not None:
            seq_name, idx = seq_name_idx
            os.makedirs(save_path_flow_unmasked, exist_ok=True)
            np.save(os.path.join(save_path_flow_unmasked, f"{seq_name}_{idx}_flow_pred.npy"), flow_pred.clone().detach().cpu().numpy())
    
    assert len(L1_error) == 1, "L1 error should be a single value."
    assert len(L2_error) == 1, "L2 error should be a single value."
    
    return {
        "L1": L1_error[0],
        "L2": L2_error[0],
        "1PE": npe_1_error["metric"],
        "3PE": npe_3_error["metric"],
    }


def main(run_path: str, jump_step: int):
    api = wandb.Api()
    run = api.run(run_path)
    config_dict = dict(run.config)
    config_dict["eval_only"] = True
    omega_config = OmegaConf.create(config_dict)

    omega_config.model.pretrain_ckpt = os.path.join("ckpt_dir", os.path.basename(run_path), "model.ckpt")
    omega_config.wandb.enabled = False
    omega_config.validation.nonrec.dataset.val.in_memory = False
    omega_config.validation.nonrec.dataset.val.force_preprocess = True
    omega_config.validation.nonrec.dataset.val.do_not_save_preprocessed = True
    omega_config.validation.nonrec.dataset.val.concat_seq = True

    print(f"Pretrained checkpoint path set to: {omega_config.model.pretrain_ckpt}")
    print(OmegaConf.to_yaml(omega_config))

    trainer = Trainer(omega_config)

    val_set = assemble_dsec_sequences(
        omega_config.dataset.common.data_root,
        include_seq=set(
            [val_seq for x in omega_config.get("validation", dict()).values() for val_seq in x.dataset.val.seq]),
        require_gt=True,
        config=omega_config.validation.nonrec.dataset.val,
        representation_type=omega_config.dataset.get("representation_type", None),
        num_bins=omega_config.dataset.get("num_voxel_bins", None)
    )

    dir_path = os.path.join("ckpt_dir", os.path.basename(run_path))
    metric_path = os.path.join(dir_path, "metrics")
    # os.makedirs(emap_path, exist_ok=True)
    
    reseults = {}

    trainer.model.eval()
    for idx in range(0, len(val_set), jump_step):
        print(f"Processing batch {idx}/{len(val_set)}")
        batch = train_collate([val_set[idx]])
        out = trainer.model(batch)

        idx_result = compute_errors(
            out["final_prediction"],
            batch["flow_gt_event_volume_new"],
            batch["flow_gt_event_volume_new_valid_mask"],
            save_path=metric_path,
            seq_name_idx=(batch["seq_name"][0], idx)
        )
        if batch["seq_name"][0] not in reseults.keys():
            reseults[batch["seq_name"][0]] = {key: [] for key in idx_result.keys()}
        for key in idx_result.keys():
            if key not in reseults[batch["seq_name"][0]]:
                reseults[batch["seq_name"][0]][key] = []
            reseults[batch["seq_name"][0]][key].append(idx_result[key])
        
    # Save results
    results_path = os.path.join(metric_path, "results.json")
    with open(results_path, 'w') as f:
        json.dump(reseults, f, indent=4)
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate optical flow predictions.")
    parser.add_argument("--run_path", type=str, required=True, help="W&B run path, e.g. 'haraghi/idnet/cmc692jl'")
    parser.add_argument("--jump_step", type=int, default=1, help="Step size for evaluation iteration")
    args = parser.parse_args()
    main(args.run_path, args.jump_step)
