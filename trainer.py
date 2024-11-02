import argparse
import functools
import glob
import gzip
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import sys
import wandb
import yaml

from abc import ABC, abstractmethod
from collections import defaultdict
from sklearn.metrics import auc
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchmetrics import ROC


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
    targets = [{'boxes': torch.tensor(item['targets']['bbox']).float(), 'labels': torch.tensor(item['targets']['labels']).long()} for item in batch]
    images = torch.tensor([item['images'] for item in batch]).float()
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


# ===
# Evaluation
# ===
def apply_nms(pred_dict):
    scores, boxes, lambda_nms = pred_dict['scores'], pred_dict['boxes'], 0.2
    selected_indices = []
    for ix, (score, box) in enumerate(zip(scores, boxes)):
        for other_box in boxes[selected_indices]:
            if intersection_over_union(box, other_box) > lambda_nms: break
        else: selected_indices.append(ix)
    return {'scores':pred_dict['scores'][selected_indices], 'boxes':pred_dict['boxes'][selected_indices], 'labels': pred_dict['labels'][selected_indices]}

    
@torch.no_grad()
def evaluate(model: nn.Module, val_loader: DataLoader, step: int, logger: Logger, class_mappings: dict[str, int]) -> None:
    model.eval()
    raise NotImplementedError('evaluate is not implemented')

    idx2class = {v:k for k, v in class_mappings.items()}
    all_predictions = []
    all_ground_truths = []
    batch_size = None
    first_batch_images = None
    for batch in val_loader:
        images, targets = batch['images'].to(device), batch['targets']
        preds = apply_nms(model(images))
        preds = [{k: v.cpu() for k, v in x.items()} for x in preds]
        all_predictions.extend(preds)

        if first_batch_images is None:
            first_batch_images = images[:batch_size]
            first_batch_preds = preds[:batch_size]
            first_batch_truths = all_ground_truths[:batch_size]
    fig_sample = plot_sample(first_batch_images, first_batch_preds, first_batch_truths, class_mappings)
    results = calculate_froc(all_predictions, all_ground_truths)
    froc_fig = plot_froc(results, step, idx2class)
    auc_froc = auc(results['average']['nlf'], results['average']['llf'])
    eval_metrics = {"auc_froc": auc_froc, 'test_images': fig_sample, 'froc_curves': froc_fig}  # Placeholder for evaluation metrics
    logger.log(eval_metrics, step)


def plot_sample(images: torch.Tensor, predictions: list[dict], ground_truths: list[dict], idx2class: dict[int, str]) -> plt.Figure:
    import matplotlib.pyplot as plt
    import torchvision.transforms as T

    transform = T.ToPILImage()
    num_images: int = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    
    if num_images == 1:
        axes = [axes]

    for idx, (img, pred, truths) in enumerate(zip(images, predictions, ground_truths)):
        ax = axes[idx]
        img = transform(img.cpu())
        ax.imshow(img)

        pred_boxes = pred['boxes']
        pred_scores = pred['scores']
        pred_labels = pred['labels']

        # Select only boxes with score >= 0.5
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            if score >= 0.5:
                xmin, ymin, xmax, ymax = box
                rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='blue', linewidth=2)
                ax.add_patch(rect)
                label_name: str = idx2class[int(label)]
                ax.text(xmin, ymin, f'{label_name}: {score:.2f}', color='red', fontsize=8, bbox=dict(facecolor='yellow', alpha=0.5))

        # Plot truth boxes
        truth_boxes = truths['boxes']
        truth_labels = truths['labels']
        for box, label in zip(truth_boxes, truth_labels):
            xmin, ymin, xmax, ymax = box
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='green', linewidth=2)
            ax.add_patch(rect)
            label_name: str = idx2class[int(label)]
            ax.text(xmin, ymin, label_name, color='green', fontsize=8, bbox=dict(facecolor='yellow', alpha=0.5))
        ax.axis('off')
    plt.tight_layout()
    return fig



def calculate_froc(predictions, truths):
    '''
    predictions: list[dict[str, torch.tensor]] => 3 Keys: boxes: torch.tensor[N, 4] (xmin, ymin, xmax, ymax), labels: torch.tensor[N], scores: torch.tensor[N]
    truths: list[dict[str, torch.tensor]] => 2 Keys: boxes: torch.tensor[N, 4] (xmin, ymin, xmax, ymax), labels: torch.tensor[N]

    Calculate FROC for each class and overall froc score for each threshold.
    # filter based on label
    # for th in thresholds
    #   apply th to filtered preds
    #   for p, l in zip(predictions, labels):
    #       for obj in l:
    #           find all the pred_dets from p with iou>0.2 with obj, select the one with highest score
    #           record it as true positive
    #       do it for all obj in l, and then the count of not matched_set is your false positive
    #   get average tp and fp
    #   record the avg tp and fp, along with th for this label
    # lastly I want to have a dict that is: {class_id: {llf: [tp_at_0.1, tp_at_0.2 ...], nlf: [fp_at_0.1, fp_at_0.2 ...]}, ..., average: {llf: [...], nlf: [...]}}
    '''
    thresholds = [x/10 for x in range(1, 11)]
    iou_threshold = 0.2
    def compute_iou(box1, box2):
        # Calculate intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        # Calculate union
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        return intersection / union if union != 0 else 0
    
    results = {class_id: {"llf": [], "nlf": []} for class_id in set(label.item() for truth in truths for label in truth['labels'])}
    results['average'] = {"llf": [], "nlf": []}
    for class_id in results.keys():
        if class_id == 'average': continue
        for th in thresholds:
            for pred, truth in zip(predictions, truths):
                pred_boxes = pred['boxes']
                pred_scores = pred['scores']
                pred_labels = pred['labels']
                
                truth_boxes = truth['boxes']
                truth_labels = truth['labels']
                
                # Filter predictions and truths based on class_id
                pred_indices = (pred_labels == class_id).nonzero(as_tuple=True)[0]
                truth_indices = (truth_labels == class_id).nonzero(as_tuple=True)[0]
                
                pred_boxes = pred_boxes[pred_indices]
                pred_scores = pred_scores[pred_indices]
                
                truth_boxes = truth_boxes[truth_indices]
                
                matched_truths = set()
                
                # Apply threshold
                th_indices = (pred_scores >= th).nonzero(as_tuple=True)[0]
                pred_boxes = pred_boxes[th_indices]
                pred_scores = pred_scores[th_indices]
                
                # Evaluate each truth object
                for truth_idx, truth_box in enumerate(truth_boxes):
                    max_iou = 0
                    best_pred_idx = -1
                    
                    # Find best matching prediction
                    for pred_idx, pred_box in enumerate(pred_boxes):
                        if pred_idx in matched_truths: continue
                        iou = compute_iou(truth_box, pred_box)
                        if iou > iou_threshold and iou > max_iou:
                            max_iou = iou
                            best_pred_idx = pred_idx
                    if best_pred_idx != -1: matched_truths.add(best_pred_idx)
                
                # Calculate false positives
                tpr.append(len(matched_truths)/len(truth_boxes) if n_targets else 0)
                fpr.append(len(pred_boxes) - len(matched_truths)/len(pred_boxes) if len(pred_boxes) else 0)
            
            avg_tp = sum(tpr) / len(tpr) if tpr else 0
            avg_fp = sum(fpr) / len(fpr) if fpr else 0
            
            results[class_id]['llf'].append(avg_tp)
            results[class_id]['nlf'].append(avg_fp)
    
    # Calculate average results
    num_classes = len(results) - 1
    for th_index in range(len(thresholds)):
        avg_llf = sum(results[class_id]['llf'][th_index] for class_id in results if class_id != 'average') / num_classes
        avg_nlf = sum(results[class_id]['nlf'][th_index] for class_id in results if class_id != 'average') / num_classes
        results['average']['llf'].append(avg_llf)
        results['average']['nlf'].append(avg_nlf)
    return results



def plot_froc(results: dict[str | int, dict[str, list[int]]], step: int, idx2class: dict[int, str]) -> plt.Figure:
    # Initialize the figure with subplots
    num_classes = len(results)
    fig, axes = plt.subplots(num_classes, 1, figsize=(10, 8 * num_classes))

    # Ensure axes is a list even if there's only one subplot
    if num_classes == 1:
        axes = [axes]

    # Plot 'average' class in the first subplot
    if 'average' in results:
        metrics = results['average']
        axes[0].plot(metrics['nlf'], metrics['llf'], label='Average')
        axes[0].set_title('Average')
        axes[0].set_xlabel('False Positives per Image (NLF)')
        axes[0].set_ylabel('True Positive Rate (LLF)')
        axes[0].grid(True)
        axes[0].legend(loc='best')

    # Plot each class starting from 0 class_id to the end
    for i, (class_id, metrics) in enumerate(results.items()):
        if class_id == 'average':
            continue
        class_name = idx2class.get(class_id, f'Class {class_id}')
        ax = axes[i]
        ax.plot(metrics['nlf'], metrics['llf'], label=class_name)
        ax.set_title(class_name)
        ax.set_xlabel('False Positives per Image (NLF)')
        ax.set_ylabel('True Positive Rate (LLF)')
        ax.grid(True)
        ax.legend(loc='best')

    # Adjust layout
    plt.tight_layout()
    return fig


# ===


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

            losses = model(images, targets)
            losses['total_loss'] = losses['loss_classifier'] + losses['loss_box_reg'] + losses['loss_objectness'] + losses['loss_rpn_box_reg']
            losses = {k: v/grad_accumulation_steps for k, v in losses.items()}
            loss = losses["total_loss"]
            loss.backward()
            for k, v in losses.items(): step_loss[k] += v.item()
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
