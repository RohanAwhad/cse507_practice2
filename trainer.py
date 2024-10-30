import argparse
import functools
import glob
import gzip
import math
import numpy as np
import os
import pickle
import random
import sys
import wandb
import yaml

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

# === Logger Classes === #
class Logger(ABC):
    @abstractmethod
    def log(self, data: dict, step: int):
        pass

class WandbLogger(Logger):
    def __init__(self, project_name, run_name, config):
        self.run = wandb.init(project=project_name, name=run_name, config=config)

    def log(self, data: dict, step: int):
        self.run.log(data, step=step)


# ===
# LR Scheduler
# ===
class CosineLRScheduler:
    def __init__(self, warmup_steps, max_steps, max_lr, min_lr):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr

    def get_lr(self, step):
        # linear warmup
        if step < self.warmup_steps:
            return self.max_lr * (step + 1) / self.warmup_steps

        # constant lr
        if step > self.max_steps:
            return self.min_lr

        # cosine annealing
        decay_ratio = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)



# === Utility to load YAML configuration === #
def load_config():
    if len(sys.argv) < 2:
        print('Please provide a config.yaml path as arg')
        exit(0)
    config_file = sys.argv[1]
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)


@functools.lru_cache(maxsize=2)
def load_shard(filepath: str) -> dict[str, np.array]:
  # Load the compressed object
  with gzip.open(filepath, 'rb') as f:
    return pickle.load(f)

class ShardedDataset(Dataset):
    def __init__(self, shard_paths: list[str], shard_size: int):
        super().__init__()
        self.shard_paths = shard_paths
        self.shard_size = shard_size
    def __len__(self): return len(self.shard_paths)*self.shard_size
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        shard_idx = index // self.shard_size
        shard = load_shard(self.shard_paths[shard_idx])
        idx = index % self.shard_size
        return {'images': shard['images'][idx], 'targets': shard['targets'][idx]}

def collate_fn(batch):
    targets = [{'boxes': torch.tensor(item['targets']['bbox_coords']), 'labels': torch.tensor(item['targets']['labels'])} for item in batch]
    images = torch.tensor([item['images'] for item in batch])
    return {'images': images, 'targets': targets}

def build_model(model_name: str, pretrained_flag: bool, num_classes: int) -> nn.Module:
    """Load or initialize a model (placeholder)."""
    # setup model
    import torchvision
    if pretrained_flag:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        # change the head
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    else:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=num_classes)
    return model

def load_dataset(dataset_path: str, shard_size: int, n_val_shards: int) -> Tuple[Dataset, Dataset]:
    all_shards = glob.glob(f"{dataset_path}/*.gz")
    all_shards = sorted(
        map(lambda x: (x, int(os.path.basename(x).split(".")[0].split("_")[1])), all_shards), key=lambda x: x[1]
    )
    random.shuffle(all_shards)
    all_shards = [x[0] for x in all_shards]
    val_shards = all_shards[:n_val_shards]
    train_shards = all_shards[n_val_shards:]
    train_dataset = ShardedDataset(shard_paths=train_shards, shard_size=shard_size)
    val_dataset = ShardedDataset(shard_paths=val_shards, shard_size=shard_size)
    return train_dataset, val_dataset


@torch.no_grad()
def evaluate(model: nn.Module, val_loader: DataLoader, step: int, logger: Logger) -> None:
    """Placeholder for evaluation logic."""
    model.eval()
    raise NotImplementedError('evaluate is not implemented')
    eval_metrics = {"accuracy": 0.9}  # Placeholder for evaluation metrics
    logger.log(eval_metrics, step)

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, ckpt_dir: str, step: int) -> None:
    """Placeholder for checkpointing logic."""
    model.eval()
    ckpt_path = os.path.join(ckpt_dir, f"checkpoint_step_{step}.pth")
    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, ckpt_path)
    print(f"Checkpoint saved at step {step}")

def next_batch(loader):
    iterator = iter(loader)
    while True:
        try:
            batch = next(iterator)
            yield batch
        except StopIteration:
            iterator = iter(loader)

# === Trainer Scaffold === #
def main():
    config = load_config()
    logger = WandbLogger(project_name=config.get('project_name', 'default_project'), run_name=config['run_name'], config=config)
    train_dataset, val_dataset = load_dataset(config['dataset_path'], config['shard_size'], config['n_val_shards'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=True,
        prefetch_factor=config['prefetch_factor'],
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=True,
        prefetch_factor=config['prefetch_factor'],
        collate_fn=collate_fn,
    )
    print('DataLoaders created')

    # TODO: remove this. just testing if loader works fine
    for batch in train_loader:
        print(batch)
        break


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = build_model(config['model_name'], config['pretrained_flag'], len(config['class_mapping']))
    model = model.to(device)
    print('Model has been loaded on device')
    # TODO: calculate save, and update where you are in the dataloader so you can restart training

    # Optimizer placeholder
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['max_lr'])
    lr_scheduler = CosineLRScheduler(config["warmup_steps"], config["max_steps"], config["max_lr"], config["min_lr"])

    # Main training loop
    step = 0
    batch_gen = next_batch(train_loader)
    grad_accumulation_steps = config["desired_batch_size"] // config["batch_size"]
    while step < config["num_steps"]:
        step += 1
        if step % config["eval_interval"] == 0:
            print(f"Step {step}: Performing evaluation")
            evaluate(model, val_loader, device, step, logger, config["class_mapping"])
        if step % config["ckpt_interval"] == 0 and config["do_ckpt"]:
            print(f"Step {step}: Saving checkpoint")
            save_checkpoint(model, os.path.join(config["ckpt_dir"], config["run_name"]), step)
        # train
        model.train()
        optimizer.zero_grad()
        step_loss = defaultdict(int)
        # gradient accumulation
        for _ in range(grad_accumulation_steps):
            batch = next(batch_gen)
            images, targets = batch['images'], batch['targets']
            images, targets = images.to(device), [{k: v.to(device) for k, v in x.items()} for x in targets]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                losses = model(images, targets)
                losses['total_loss'] = losses['loss_classifier'] + losses['loss_box_reg'] + losses['loss_objectness'] + losses['loss_rpn_box_reg']
                losses = {k: v/grad_accumulation_steps for k, v in losses.items()}
                loss = losses["total_loss"]
            loss.backward()
            for k, v in losses: step_loss[k] += v.item()
        lr = lr_scheduler.get_lr(step)
        for param_group in optimizer.param_groups: param_group["lr"] = lr
        optimizer.step()
        log_data = dict(train_loss=step_loss, lr=lr)
        logger.log(log_data, step)

        if step >= config["num_steps"]: break
    batch_gen.close()
    logger.log({"status": "Training finished"}, step=step)

if __name__ == "__main__":
    main()
