# main_script.py

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import flow_vis
import imageio

import wandb
from omegaconf import OmegaConf

from idn.utils.trainer import Trainer
from idn.utils.validation import Validator
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

def visualize_and_save_flow(flow_pred, flow_gt, valid_mask, save_path=None, seq_name=None):
    valid_mask = valid_mask.astype(bool)
    masked_pred = np.zeros_like(flow_pred)
    masked_gt = np.zeros_like(flow_gt)
    masked_pred[:, valid_mask] = flow_pred[:, valid_mask]
    masked_gt[:, valid_mask] = flow_gt[:, valid_mask]

    masked_pred_vis = np.transpose(masked_pred, (1, 2, 0))
    masked_gt_vis = np.transpose(masked_gt, (1, 2, 0))

    pred_img = flow_vis.flow_to_color(masked_pred_vis, convert_to_bgr=False)
    gt_img = flow_vis.flow_to_color(masked_gt_vis, convert_to_bgr=False)

    combined = np.concatenate([gt_img, pred_img], axis=1)

    plt.figure(figsize=(12, 6))
    plt.imshow(combined)
    plt.axis('off')
    plt.title("Left: Ground Truth Flow | Right: Predicted Flow")
    plt.tight_layout()
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, seq_name + "_flow_comparison.png"), bbox_inches='tight', pad_inches=0)
        imageio.imwrite(os.path.join(save_path, seq_name + "_flow_pred.png"), pred_img)
        imageio.imwrite(os.path.join(save_path, seq_name + "_flow_gt.png"), gt_img)
    plt.close()
    print(f"Saved flow comparison image to {save_path}")

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
    os.makedirs(os.path.join(dir_path, "flow_viz"), exist_ok=True)

    trainer.model.eval()
    for idx in range(0, len(val_set), jump_step):
        print(f"Processing batch {idx}/{len(val_set)}")
        batch = train_collate([val_set[idx]])
        out = trainer.model(batch)

        flow_pred = out["final_prediction"][0].clone().detach().cpu().numpy()
        flow_gt = batch["flow_gt_event_volume_new"][0].clone().detach().cpu().numpy()
        valid_mask = batch["flow_gt_event_volume_new_valid_mask"][0, 0].clone().detach().cpu().numpy()

        visualize_and_save_flow(
            flow_pred, flow_gt, valid_mask,
            os.path.join(dir_path, "flow_viz"),
            batch["seq_name"][0] + f"_{idx:04d}"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and visualize optical flow predictions.")
    parser.add_argument("--run_path", type=str, required=True, help="W&B run path, e.g. 'haraghi/idnet/cmc692jl'")
    parser.add_argument("--jump_step", type=int, default=20, help="Step size for evaluation iteration")
    args = parser.parse_args()
    main(args.run_path, args.jump_step)
