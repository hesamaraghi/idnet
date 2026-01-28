from omegaconf import OmegaConf
import hydra
from .utils.trainer import Trainer

# @hydra.main(config_path="config", config_name="mvsec_train")
# @hydra.main(config_path="config", config_name="tid_train")
@hydra.main(config_path="config", config_name="id_train_eigen")

def main(config):
    
    cmd_cfg = OmegaConf.from_cli()
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
