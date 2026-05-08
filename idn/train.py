from omegaconf import OmegaConf
import hydra
import sys
from .utils.trainer import Trainer


def cli_overrides_without_hydra_flags():
    hydra_flags_with_values = {
        "--config-name",
        "--config-path",
        "--config-dir",
        "--cfg",
        "--package",
    }
    hydra_flags_without_values = {
        "--run",
        "--multirun",
        "-m",
        "--help",
        "-h",
        "--info",
    }
    overrides = []
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        flag = arg.split("=", 1)[0]
        if flag in hydra_flags_with_values:
            skip_next = "=" not in arg
            continue
        if flag in hydra_flags_without_values:
            continue
        if arg.startswith("+"):
            continue
        if arg.startswith("-"):
            continue
        overrides.append(arg)
    return OmegaConf.from_cli(overrides)


# @hydra.main(config_path="config", config_name="mvsec_train")
# @hydra.main(config_path="config", config_name="tid_train")
@hydra.main(config_path="config", config_name="id_train_eigen")

def main(config):
    
    cmd_cfg = cli_overrides_without_hydra_flags()
    config = OmegaConf.merge(config, cmd_cfg)
    
    if config.get("resume_ckpt", None):
        print(f"Try reading config from wandb server for resume...")
        try:
            import wandb
            api = wandb.Api()
            # extract run path from resume checkpoint path
            run_path = "/".join(["haraghi",config.wandb.project, config.resume_ckpt.split("/")[-2]])
            run = api.run(run_path)
            config_dict = dict(run.config)
            wandb_omega_config = OmegaConf.create(config_dict)
            wandb_omega_config.resume_ckpt = config.resume_ckpt
            config = OmegaConf.merge(config, wandb_omega_config)
            print(f"Successfully read config from wandb server for resume.")
        except Exception as e:
            print(f"Failed to read config from wandb server for resume. Exception: {e}")
            
    
    config = OmegaConf.merge(config, cmd_cfg)
    print(OmegaConf.to_yaml(config))

    trainer = Trainer(config)

    print("Number of parameters: ", sum(p.numel()
          for p in trainer.model.parameters() if p.requires_grad))

    trainer.fit()


if __name__ == '__main__':
    main()
