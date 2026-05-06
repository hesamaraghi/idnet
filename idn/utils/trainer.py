import os
import json
import wandb
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset
from tqdm import tqdm
from types import GeneratorType
from collections import namedtuple
from torchinfo import summary
from torch.optim.lr_scheduler import OneCycleLR
from omegaconf import OmegaConf

from .torch_environ import config_torch
from .helper_functions import move_batch_to_cuda
from .model_utils import get_model_by_name
from .loss_utils import compute_seq_loss, get_loss_fn_by_name
from .validation import Validator
from .callbacks import CallbackBridge
from .exp_tracker import ExpTracker
from .retrieval_fn import get_retreival_fn
from ..loader.loader_dsec import (
    Sequence,
    RepresentationType,
    DatasetProvider,
    assemble_dsec_sequences,
    assemble_dsec_test_set,
    train_collate,
    rec_train_collate
)
from ..loader.loader_mvsec import (
    MVSEC,
    MVSECRecurrent
)


def load_toy_dataset_metadata(data_root: str, seq_name: str):
    """Load dataset metadata from JSON file for toy datasets.
    
    Args:
        data_root: Root directory (e.g., 'toy_datasets/data/star8')
        seq_name: Sequence name (e.g., 'star8' or 'star8_test')
        
    Returns:
        dict: Metadata dictionary with generation parameters, or None if not found
    """
    metadata_path = os.path.join(data_root, "train_optical_flow", seq_name, "dataset_metadata.json")
    if not os.path.exists(metadata_path):
        return None
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        print(f"Warning: Failed to load metadata from {metadata_path}: {e}")
        return None


class Trainer(CallbackBridge):
    def __init__(self, config, model=None):
        super().__init__()
        self.config = config
        config_torch(config.torch)
        
        # TODO: it should apply to torch not just like this
        # Set random seed if provided via environment variable (for multi-seed experiments)
        if 'TRAINING_SEED' in os.environ:
            seed = int(os.environ['TRAINING_SEED'])
            print(f"Setting random seed to {seed} for reproducibility")
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            import numpy as np
            import random
            np.random.seed(seed)
            random.seed(seed)
            # Note: For full determinism, also set torch.backends.cudnn.deterministic = True
            # but this may impact performance
        
        if config.get("wandb", {}).get("enabled", False):
            print("Initializing Weights & Biases logging...")
            print("Project:", config.wandb.project)
            print("Run name:", config.wandb.get("run_name", None))

            wandb.init(
                project=config.wandb.project,
                name=config.wandb.get("run_name", None),
                config=OmegaConf.to_object(config),  # logs your full config
                resume="allow",
            )
            
            # Log dataset generation parameters from environment if available
            # These are set by the generate_and_train.py wrapper for sweeps
            if 'DATASET_GEN_SAVE_STEP' in os.environ:
                wandb.config.update({
                    'dataset_gen_save_step': int(os.environ['DATASET_GEN_SAVE_STEP']),
                    'dataset_gen_total_frames': int(os.environ['DATASET_GEN_TOTAL_FRAMES']),
                    'dataset_gen_test_size': float(os.environ['DATASET_GEN_TEST_SIZE']),
                    'dataset_gen_variant_hash': os.environ['DATASET_GEN_VARIANT_HASH'],
                }, allow_val_change=True)
                print(f"✓ Logged dataset generation params to wandb:")
                print(f"   save_step={os.environ['DATASET_GEN_SAVE_STEP']}")
                print(f"   total_frames={os.environ['DATASET_GEN_TOTAL_FRAMES']}")
                print(f"   variant_hash={os.environ['DATASET_GEN_VARIANT_HASH']}")
            
            #TODO: it should go through config and not with envs
            # Log training seed if it was set
            if 'TRAINING_SEED' in os.environ:
                wandb.config.update({
                    'training_seed': int(os.environ['TRAINING_SEED']),
                }, allow_val_change=True)
                print(f"✓ Logged training seed to wandb: {os.environ['TRAINING_SEED']}")
            
            # Load and log dataset metadata if available
            # Collect all sequence names from train and val
            train_seqs = []
            val_seqs = []
            
            # Check if we're using all sequences
            use_all_seqs = False
            
            # Collect sequences from validation config
            for val_name, val_cfg in config.get("validation", {}).items():
                if val_cfg and hasattr(val_cfg, "dataset"):
                    if hasattr(val_cfg.dataset, "train"):
                        # Check if use_all_seqs is enabled
                        if hasattr(val_cfg.dataset.train, "use_all_seqs") and val_cfg.dataset.train.use_all_seqs:
                            use_all_seqs = True
                        elif hasattr(val_cfg.dataset.train, "seq"):
                            train_seqs.extend(val_cfg.dataset.train.seq)
                    
                    if hasattr(val_cfg.dataset, "val") and hasattr(val_cfg.dataset.val, "seq"):
                        val_seqs.extend(val_cfg.dataset.val.seq)
            
            # If use_all_seqs is enabled, scan the directory for all sequences
            if use_all_seqs:
                try:
                    flow_gt_root = os.path.join(config.dataset.common.data_root, "train_optical_flow")
                    if os.path.exists(flow_gt_root):
                        all_seqs = os.listdir(flow_gt_root)
                        # Filter out non-directory entries
                        all_seqs = [seq for seq in all_seqs if os.path.isdir(os.path.join(flow_gt_root, seq))]
                        train_seqs.extend(all_seqs)
                        print(f"✓ Detected use_all_seqs=True, found sequences: {all_seqs}")
                except Exception as e:
                    print(f"Warning: Could not scan directory for sequences: {e}")
            
            # Try to load metadata for each sequence
            dataset_metadata = {}
            for seq_name in set(train_seqs + val_seqs):
                metadata = load_toy_dataset_metadata(
                    config.dataset.common.data_root, 
                    seq_name
                )
                if metadata:
                    dataset_metadata[f"dataset_metadata_{seq_name}"] = metadata
                    print(f"✓ Loaded dataset metadata for sequence: {seq_name}")
                    print(f"   save_step: {metadata.get('save_step')}, "
                          f"total_frames: {metadata.get('total_frames')}, "
                          f"image_size: {metadata.get('image_size')}")
            
            # Log metadata to wandb if any was found
            if dataset_metadata:
                wandb.config.update(dataset_metadata, allow_val_change=True)
                print(f"✓ Logged metadata for {len(dataset_metadata)} sequences to Wandb")
        
        self.model = model if model is not None else \
            get_model_by_name(config.model.name, config.model)
        if config.model.get("pretrain_ckpt", None):
            self.resume_model_from_ckpt(config.model.pretrain_ckpt)

        if not self.config.get("eval_only", False):
            self.train_dataloader = self.configure_train_dataloader()
            self.configure_loss()
            self.optimizer = self.configure_optimizer()
            if config.optim.get("scheduler", None):
                self.scheduler = OneCycleLR(
                    self.optimizer, max_lr=self.config.optim.lr,
                    steps_per_epoch=len(self.train_dataloader), 
                    epochs=self.config.num_epoch,
                    pct_start=0.05, cycle_momentum=False, anneal_strategy='linear',
                )
            else:
                self.scheduler = None

            self.epoch = 0
            self.step = 0
            self.batches_seen = 0
            self.samples_seen = 0
            if config.get("resume_ckpt", None):
                resume_only_model = config.get("finetune", False)
                self.resume_from_ckpt(config.resume_ckpt, resume_only_model)
            self.logger = self.configure_tracker()
        
            self.configure_callbacks(config.callbacks)

            self.execute_callbacks("on_init_end")
        
            if config.get("wandb", {}).get("enabled", False):
                wandb.config.update(OmegaConf.to_object(config))

    def configure_train_dataloader(self):
        if self.config.dataset.dataset_name == "dsec":
            train_set = assemble_dsec_sequences(
                self.config.dataset.common.data_root,
                include_seq=set(
                    [val_seq for x in self.config.get("validation", dict()).values() for val_seq in x.dataset.train.seq]),
                exclude_seq=set(
                    [val_seq for x in self.config.get("validation", dict()).values() for val_seq in x.dataset.val.seq]) \
                    if self.config.dataset.train.exclude_val else None,
                require_gt=True,
                config=self.config.dataset.train,
                representation_type=self.config.dataset.get("representation_type", None),
                num_bins=self.config.dataset.get("num_voxel_bins", None)
            )

        elif self.config.dataset.dataset_name == "mvsec":
            train_set = MVSEC(config=self.config.dataset, training=True) #20Hz
            train_set = self.configure_mvsec_subset(train_set)
        elif self.config.dataset.dataset_name == "mvsec_recurrent":
            train_set = MVSECRecurrent("outdoor_day2", augment=False, 
                                       sequence_length=self.config.dataset.train.sequence_length)
        else:
            raise NotImplementedError
        collate_fn = rec_train_collate \
            if hasattr(self.config.dataset.train, "recurrent") \
            and self.config.dataset.train.recurrent else train_collate
        return DataLoader(
            train_set, collate_fn=collate_fn, **self.config.data_loader.train.args)

    def configure_mvsec_subset(self, train_set):
        subset_cfg = self.config.dataset.train.get("subset", None)
        if subset_cfg is None or not subset_cfg.get("enabled", False):
            return train_set

        n = len(train_set)
        if n <= 0:
            raise ValueError("Cannot subset an empty MVSEC training dataset")

        mode = subset_cfg.get("mode", "uniform")
        if mode == "range":
            start = int(subset_cfg.get("start", 0))
            end_value = subset_cfg.get("end", None)
            end = n if end_value is None else int(end_value)
            stride = int(subset_cfg.get("stride", 1))
            if stride <= 0:
                raise ValueError(f"MVSEC subset stride must be positive, got {stride}")
            start = max(start, 0)
            end = min(end, n)
            indices = list(range(start, end, stride))
        else:
            count = subset_cfg.get("count", None)
            fraction = subset_cfg.get("fraction", None)
            if count is None:
                if fraction is None:
                    raise ValueError("MVSEC subset needs either count or fraction")
                fraction = float(fraction)
                if fraction <= 0:
                    raise ValueError(f"MVSEC subset fraction must be positive, got {fraction}")
                count = round(n * fraction)
            count = int(count)
            if count <= 0:
                raise ValueError(f"MVSEC subset count must be positive, got {count}")
            count = min(count, n)

            if mode == "uniform":
                if count == n:
                    indices = list(range(n))
                else:
                    indices = torch.linspace(0, n - 1, steps=count).round().long().unique().tolist()
            elif mode == "random":
                seed = int(subset_cfg.get("seed", 0))
                generator = torch.Generator().manual_seed(seed)
                indices = torch.randperm(n, generator=generator)[:count].sort().values.tolist()
            else:
                raise ValueError(f"Unknown MVSEC subset mode: {mode}")

        if not indices:
            raise ValueError("MVSEC subset produced no indices")

        print(
            f"Using MVSEC training subset: mode={mode}, "
            f"samples={len(indices)}/{n}, first={indices[0]}, last={indices[-1]}",
            flush=True,
        )
        return Subset(train_set, indices)

    def configure_optimizer(self):
        if self.config.optim.optimizer == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.config.optim.lr)
        elif self.config.optim.optimizer == "adamw":
            return torch.optim.AdamW(self.model.parameters(), lr=self.config.optim.lr)
        else:
            raise NotImplementedError

    def resume_model_from_ckpt(self, ckpt):
        ckpt = torch.load(ckpt, weights_only=False)
        if "model" in ckpt:
            self.model.load_state_dict(ckpt["model"])
        elif "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
        else:
            try:
                self.model.load_state_dict(ckpt)
            except:
                raise ValueError("Invalid checkpoint")
    
    def resume_from_ckpt(self, ckpt, resume_only_model=False):
        print(f"Resuming from checkpoint: {ckpt}, resume_only_model={resume_only_model}")
        ckpt = torch.load(ckpt, map_location='cpu', weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        if not resume_only_model:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.epoch = ckpt['epoch']
            if self.scheduler and 'scheduler_state_dict' in ckpt:
                self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if "tracker" in ckpt:
            self.logged_tracker = ckpt['tracker']

    def configure_tracker(self):
        return ExpTracker()

    def configure_loss(self):
        lc = namedtuple("loss_config",
                        ["retrieval_fn", "loss_fn", "weight", "seq_weight", "seq_norm"])
        self.loss_config = dict()
        for quantity, config in self.config.loss.items():
            self.loss_config[quantity] = lc(
                get_retreival_fn(quantity),
                get_loss_fn_by_name(config.loss_type),
                config.get("weight", 1.0),
                config.get("seq_weight", None),
                config.get("seq_norm", False))

    def train_epoch(self):
        self.execute_callbacks("on_epoch_begin")
        self.model.train()
        self.model.cuda(self.config.data_loader.train.gpu)
        # This is necessary to sync optimizer parameter device with model device
        self.optimizer.load_state_dict(self.optimizer.state_dict())
        for batch in tqdm(self.train_dataloader):
            self.execute_callbacks("on_batch_begin")
            self.optimizer.zero_grad()
            batch = move_batch_to_cuda(
                batch, self.config.data_loader.train.gpu)
            out = self.model(batch)
            if isinstance(out, GeneratorType):
                loss_item = []
                for i, ret in enumerate(out):
                    seq_len = len(ret["flow_trajectory"])
                    loss, loss_breakdown = self.compute_loss(
                        ret, batch[i*seq_len:(i+1)*seq_len])
                    loss.backward()
                    loss_item.append(loss.detach().item())
                self.loss = sum(loss_item)/len(loss_item)
                self.loss_1 = \
                loss_item[0]
                for loss_type, l in loss_breakdown.items():
                    setattr(self, 'loss_'+loss_type, l.item())

            else:
                self.loss, _ = self.compute_loss(out, batch)
                self.loss.backward()
            self.execute_callbacks("on_step_begin")
            self.optimizer.step()
            self.execute_callbacks("on_step_end")
            self.execute_callbacks("on_batch_end")
            self.step += 1
            if self.scheduler:
                self.scheduler.step()
                self.lr = self.scheduler.get_last_lr()[0]
        self.execute_callbacks("on_epoch_end")
        self.epoch += 1

    def compute_loss(self, ret, batch):
        loss = dict()
        total_loss = 0
        for quantity, config in self.loss_config.items():
            estimate, ground_truth = config.retrieval_fn(ret, batch)
            if isinstance(estimate, list):
                loss_fn = lambda estimate, ground_truth: \
                    compute_seq_loss(config.seq_weight, config.loss_fn,
                                     estimate, ground_truth)
            else:
                loss_fn = config.loss_fn
            
            loss[quantity] = loss_fn(estimate, ground_truth)
            total_loss += config.weight * loss[quantity]
        return total_loss, loss

    def fit(self, epochs=None):
        num_epochs = epochs if epochs is not None else self.config.num_epoch
        self.execute_callbacks("on_train_begin")
        try:
            while self.epoch < num_epochs:
                self.train_epoch()
                print(f"Epoch {self.epoch} finished, loss: {self.loss:.4f}", flush=True)
        except:
            raise Exception("Training failed")
        finally:
            self.execute_callbacks("on_train_end")
