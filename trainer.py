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
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
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
        config = yaml.safe_load(file)
    if len(sys.argv) > 2:
        print("Setting run name:", sys.argv[2])
        config["run_name"] = sys.argv[2]
    return config


@functools.lru_cache(maxsize=3)
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
    images = torch.tensor(np.array([item['images'] for item in batch])).float()
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
            if compute_iou(box, other_box) > lambda_nms: break
        else: selected_indices.append(ix)
    return {'scores':pred_dict['scores'][selected_indices], 'boxes':pred_dict['boxes'][selected_indices], 'labels': pred_dict['labels'][selected_indices]}

    
@torch.no_grad()
def evaluate(model: nn.Module, val_loader: DataLoader, device, step: int, logger: Logger, class_mappings: dict[str, int]) -> None:
    model.eval()
    idx2class = {v:k for k, v in class_mappings.items()}
    all_predictions = []
    all_ground_truths = []
    first_batch_images, first_batch_preds, first_batch_truths = None, None, None
    for batch in tqdm(val_loader, total=len(val_loader), desc='Validation'):
        images, targets = batch['images'].to(device), batch['targets']
        preds = model(images)
        preds = [{k: v.cpu() for k, v in apply_nms(x).items()} for x in preds]
        all_predictions.extend(preds)
        all_ground_truths.extend(targets)
        if first_batch_images is None:
            first_batch_images = images
            first_batch_preds = preds
            first_batch_truths = targets
    fig_sample = plot_sample(first_batch_images, first_batch_preds, first_batch_truths, idx2class)
    results = calculate_froc(all_predictions, all_ground_truths, len(class_mappings))
    froc_fig = plot_froc(results, step, idx2class)
    auc_froc = auc(results['average']['nlf'], results['average']['llf'])

    # transform results for wandb
    new_results = {}
    for class_id, v in results.items():
        class_label = idx2class[class_id] if class_id != 'average' else 'average'
        new_results[class_label] = {}
        for i, (l, n) in enumerate(zip(v['llf'], v['nlf'])): new_results[class_label][(i+1)/10] = {'llf': l, 'nlf': n}

    eval_metrics, sample_images, froc_curves = {"auc_froc": auc_froc, 'all_results_from_val': new_results}, {'test_images': fig_sample}, {'froc_curves': froc_fig}  # Placeholder for evaluation metrics
    logger.log(eval_metrics, step)
    logger.log(sample_images, step)
    logger.log(froc_curves, step)


# Define the improved plot_sample function for visualization
def plot_sample(images, predictions, ground_truths, idx2class):
    import matplotlib.pyplot as plt
    import torchvision.transforms as T

    transform = T.ToPILImage()
    num_images = len(images)
    fig_width = max(18, 6 * num_images)  # Adjust figure width for optimal spacing
    fig, axes = plt.subplots(1, num_images, figsize=(fig_width, 7))
    
    # Ensure axes is always iterable
    if num_images == 1:
        axes = [axes]
    
    for idx, (img, pred, truths) in enumerate(zip(images, predictions, ground_truths)):
        ax = axes[idx]
        
        # Convert image to PIL format for plotting
        img_pil = transform(img.cpu())
        ax.imshow(img_pil)  # Use grayscale for clear contrast
        
        # Plot predicted boxes with a threshold score for improved filtering
        for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
            if score >= 0.5:  # Higher threshold to show confident predictions only
                xmin, ymin, xmax, ymax = box
                rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                     edgecolor='blue', linewidth=2.5, fill=False)
                ax.add_patch(rect)
                
                label_name = idx2class[int(label)]
                ax.text(xmin, ymin - 2, f'{label_name}: {score:.2f}', 
                        color='white', fontsize=9, fontweight='bold',
                        bbox=dict(facecolor='blue', alpha=0.7, edgecolor='none', pad=1))

        # Plot ground truth boxes with distinctive color and position
        for box, label in zip(truths['boxes'], truths['labels']):
            xmin, ymin, xmax, ymax = box
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                 edgecolor='lime', linewidth=2.5, fill=False)
            ax.add_patch(rect)
            
            label_name = idx2class[int(label)]
            ax.text(xmin, ymax + 5, label_name, 
                    color='black', fontsize=9, fontweight='bold',
                    bbox=dict(facecolor='lime', alpha=0.7, edgecolor='none', pad=1))

        # Remove axis for a cleaner look
        ax.axis('off')
    
    plt.subplots_adjust(wspace=0.1)  # Reduce whitespace between images
    plt.tight_layout()
    return fig

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



def calculate_froc(predictions: List[Dict[str, torch.Tensor]], truths: List[Dict[str, torch.Tensor]], num_classes: int) -> Dict[Union[int, str], Dict[str, List[float]]]:
    thresholds = [x / 10 for x in range(1, 10)]
    iou_threshold: float = 0.2  # as per Grand Challenge

    results: Dict[Union[int, str], Dict[str, List[float]]] = {
        class_id: {"llf": [], "nlf": []} for class_id in set(label.item() for truth in truths for label in truth['labels'])
    }
    results['average'] = {"llf": [], "nlf": []}

    for class_id in results.keys():
        if class_id == 'average':
            continue
        
        for th in thresholds:
            tpr: List[float] = []
            fpr: List[float] = []
            
            for pred, truth in zip(predictions, truths):
                pred_boxes = pred['boxes']
                pred_scores = pred['scores']
                pred_labels = pred['labels']
                
                truth_boxes = truth['boxes']
                truth_labels = truth['labels']
                
                pred_indices = (pred_labels == class_id).nonzero(as_tuple=True)[0]
                truth_indices = (truth_labels == class_id).nonzero(as_tuple=True)[0]
                
                pred_boxes = pred_boxes[pred_indices]
                pred_scores = pred_scores[pred_indices]
                
                truth_boxes = truth_boxes[truth_indices]
                
                matched_truths = set()
                
                th_indices = (pred_scores >= th).nonzero(as_tuple=True)[0]
                pred_boxes = pred_boxes[th_indices]
                pred_scores = pred_scores[th_indices]

                for truth_idx, truth_box in enumerate(truth_boxes):
                    max_iou = 0
                    best_pred_idx = -1
                    
                    for pred_idx, pred_box in enumerate(pred_boxes):
                        if pred_idx in matched_truths:
                            continue
                        iou = compute_iou(truth_box, pred_box)
                        if iou > iou_threshold and iou > max_iou:
                            max_iou = iou
                            best_pred_idx = pred_idx
                    if best_pred_idx != -1:
                        matched_truths.add(best_pred_idx)

                n_targets = len(truth_boxes)
                tpr.append(len(matched_truths) / n_targets if n_targets else 0)
                fpr.append((len(pred_boxes) - len(matched_truths)) / len(pred_boxes) if len(pred_boxes) else 0)

            avg_tp = sum(tpr) / len(tpr) if tpr else 0
            avg_fp = sum(fpr) / len(fpr) if fpr else 0
            
            results[class_id]['llf'].append(avg_tp)
            results[class_id]['nlf'].append(avg_fp)

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
        ax = axes[i+1]
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
    os.makedirs(ckpt_dir, exist_ok=True)
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
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=True,
        prefetch_factor=config['prefetch_factor'],
        collate_fn=collate_fn,
        persistent_workers=True,
    )
    print('DataLoaders created')
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
            save_checkpoint(model, optimizer, os.path.join(config["ckpt_dir"], config["run_name"]), step)
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

        # Gradient clipping
        max_grad_norm = config.get('max_grad_norm', None)
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        # get learning rate
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
