import argparse
import dataclasses
import json
import os
import pandas as pd
import warnings
from tqdm import tqdm
from typing import Callable, Dict, List
import pickle
import gzip

# ===
# Constants
# ===
batch_size = 2
CSV_FILEPATH = './tiny_ds/tiny_train.csv'
DICOM_PATH = './tiny_ds/ds'
SHARD_DIR = './tiny_ds/shards'


# ===
# Localization Dataset
# ===
import torch
from torch.utils.data import Dataset, DataLoader
import pydicom
import cv2
from PIL import Image
import numpy as np

TRAIN_IMAGE_HEIGHT = 512
TRAIN_IMAGE_WIDTH = 512

class LocalizationVinDrDS(Dataset):
  def __init__(self, csv_filepath, dicom_dir):
    super().__init__()
    self.dicom_dir = dicom_dir
    dicom_files = {f.split('.')[0] for f in os.listdir(self.dicom_dir) if f.endswith('.dicom')}

    self.df = pd.read_csv(csv_filepath)
    self.df = self.df[self.df['image_id'].isin(dicom_files)]

  def __len__(self): return len(self.df)
  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    image_id = row['image_id']
    dicom_path = os.path.join(self.dicom_dir, f"{image_id}.dicom")
    
    # Load the DICOM image
    dicom_image = pydicom.dcmread(dicom_path).pixel_array
    image: np.ndarray = cv2.resize(dicom_image, (TRAIN_IMAGE_WIDTH, TRAIN_IMAGE_HEIGHT))
    
    # Normalize bounding box coordinates
    orig_width, orig_height = dicom_image.shape
    xmin: float = row.get('x_min', 0) / orig_width
    ymin: float = row.get('y_min', 0) / orig_height
    xmax: float = row.get('x_max', orig_width) / orig_width
    ymax: float = row.get('y_max', orig_height) / orig_height

    return {
      "images": torch.tensor(image),
      "labels": torch.tensor((row['class_id'], )),
      "bbox_coords": torch.tensor((xmin, ymin, xmax, ymax))
    }

dataset = LocalizationVinDrDS(CSV_FILEPATH, DICOM_PATH)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True, prefetch_factor=64, drop_last=True)


# ===
# Run (ideally nothing would change below this)
# ===
def save_shard(obj: dict[str, np.array], filepath: str) -> None:
  # Serialize the object and compress it using gzip
  with gzip.open(filepath, 'wb') as f:
    pickle.dump(obj, f)

def load_shard(filepath: str) -> dict[str, np.array]:
  # Load the compressed object
  with gzip.open(filepath, 'rb') as f:
    return pickle.load(f)

os.makedirs(SHARD_DIR, exist_ok=True)
shard_count = 0

# TODO: use dynamic key names? they are repeated quite a lot
curr_shard = {'images': [], 'labels': [], 'bbox_coords': []}
for batch in tqdm(dataloader, total=len(dataloader), desc="Saving shards"):
  curr_shard['images'].append(batch['images'].numpy())
  curr_shard['labels'].append(batch['labels'].numpy())
  curr_shard['bbox_coords'].append(batch['bbox_coords'].numpy())

  if len(curr_shard['images']) >= 1024//batch_size:
    shard_path = os.path.join(SHARD_DIR, f"shard_{shard_count:04d}.gz")
    final_shard = {
      'images': np.concatenate(curr_shard['images'], axis=0),
      'labels': np.concatenate(curr_shard['labels'], axis=0),
      'bbox_coords': np.concatenate(curr_shard['bbox_coords'], axis=0),
    }
    save_shard(final_shard, shard_path)
    shard_count += 1
    curr_shard = {'images': [], 'labels': [], 'bbox_coords': []}

if len(curr_shard['images']) > 0:
  shard_path = os.path.join(SHARD_DIR, f"shard_{shard_count:04d}.gz")
  final_shard = {
    'images': np.concatenate(curr_shard['images'], axis=0),
    'labels': np.concatenate(curr_shard['labels'], axis=0),
    'bbox_coords': np.concatenate(curr_shard['bbox_coords'], axis=0),
  }
  save_shard(final_shard, shard_path)
