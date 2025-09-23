import argparse
from omegaconf import OmegaConf
import hydra
from hydra import initialize, compose
from .utils.trainer import Trainer
from .utils.validation import Validator
import wandb
import os

def test(trainer, omega_config, save_dir):
    with initialize(config_path="config"):
        test_cfg = compose(config_name="validation/dsec_test",
                        overrides=[]).validation
    test_cfg.dataset = omega_config.dataset
    test_cfg.logger.save_dir = save_dir
    Validator.get_test_type("dsec")(test_cfg).execute_test(
        trainer.model, save_all=False)
    

def test_co(trainer):
    test_cfg = compose(config_name="validation/dsec_co",
                       overrides=[]).validation
    Validator.get_test_type("dsec", "co")(
        test_cfg).execute_test(trainer.model, save_all=False)


def main(args):
    api = wandb.Api()
    run = api.run(args.run_path)
    config_dict = dict(run.config)
    config_dict["eval_only"] = True
    omega_config = OmegaConf.create(config_dict)
    
    omega_config.model.pretrain_ckpt = os.path.join("ckpt_dir", os.path.basename(args.run_path), "model.ckpt")
    omega_config.wandb.enabled = False
    omega_config.dataset.val.in_memory = False
    omega_config.dataset.train.in_memory = omega_config.dataset.val.in_memory
    omega_config.dataset.val.force_preprocess = False
    omega_config.dataset.train.force_preprocess = omega_config.dataset.val.force_preprocess
    omega_config.dataset.val.do_not_save_preprocessed = True
    omega_config.dataset.train.do_not_save_preprocessed = omega_config.dataset.val.do_not_save_preprocessed
    omega_config.dataset.val.concat_seq = False
    if args.from_tmp:
        omega_config.dataset.common.data_root = "/tmp/maraghi/datasets/DSEC/"
        omega_config.dataset.common.test_root = "/tmp/maraghi/datasets/DSEC/test/"
    else:
        omega_config.dataset.common.data_root = "data/"
        omega_config.dataset.common.test_root = "data/test/"
    save_dir = os.path.join(args.save_dir, os.path.basename(args.run_path), "eval")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    print(f"Pretrained checkpoint path set to: {omega_config.model.pretrain_ckpt}")
    print(OmegaConf.to_yaml(omega_config))

    trainer = Trainer(omega_config)

    print("Number of parameters: ", sum(p.numel()
          for p in trainer.model.parameters() if p.requires_grad))


    if omega_config.model.name == "RecIDE":
        test_co(trainer)
    elif omega_config.model.name == "IDEDEQIDO":
        test(trainer, omega_config, save_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform Test.")
    parser.add_argument("--run_path", type=str, required=True, help="W&B run path, e.g. 'haraghi/idnet/cmc692jl'")
    parser.add_argument("--save_dir", type=str, default="ckpt_dir", help="Directory to save checkpoints")
    parser.add_argument("--from_tmp", action='store_true', help="Use this flag if running from a temporary directory")
    args = parser.parse_args()
    main(args)
