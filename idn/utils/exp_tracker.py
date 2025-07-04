import os
import wandb
class ExpTracker:
    def __init__(self) -> None:
        self.log_dir = os.path.join("ckpt_dir",wandb.run.id)  if wandb.run else None
        if self.log_dir:
            print(f"Logging to directory: {self.log_dir}")
            os.makedirs(self.log_dir, exist_ok=True)

    def on_init_end(self, *args, **kwargs):
        pass

    def on_exp_begin(self, *args, **kwargs):
        pass

    def log_dict_at_step(self, dict, step=None):
        if wandb.run:
            flattened = self.flatten_dict(dict)
            wandb.log(flattened, step=step)

    def flatten_dict(self, d, parent_key="", sep="/"):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            else:
                # Convert torch.Tensor to float if necessary
                if hasattr(v, "item"):
                    try:
                        v = v.item()
                    except:
                        pass
                items.append((new_key, v))
        return dict(items)

    def summary(self):
        return {
            "id": "exp_tracker",
        }
