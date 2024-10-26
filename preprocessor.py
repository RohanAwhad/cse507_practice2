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
SHARD_SIZE = 1024
batch_size = 1  # because dataloader is not compatible with localization labels
CSV_FILEPATH = '/data/courses/2024/class_ImageSummerFall2024_jliang12/vinbigdata/train.csv'
DICOM_PATH = '/data/courses/2024/class_ImageSummerFall2024_jliang12/vinbigdata/train'
SHARD_DIR = '/scratch/rawhad/CSE507/practice_2_v2/shards'


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
  def __init__(self, csv_filepath: str, dicom_dir: str) -> None:
    super().__init__()
    self.dicom_dir = dicom_dir
    dicom_files = {f.split('.')[0] for f in os.listdir(self.dicom_dir) if f.endswith('.dicom')}
    self.df = pd.read_csv(csv_filepath)
    self.df = self.df[self.df['image_id'].isin(dicom_files)]
    # because faster rcnn expects 0 as no finding class id and we have 14 as no finding class, following steps are taken
    self.df['class_id'] += 1
    self.df.loc[self.df['class_id'] == 15, 'class_id'] = 0
    self.unique_ids = self.df['image_id'].unique().tolist()

  def __len__(self) -> int:
    return len(self.unique_ids)
  
  def __getitem__(self, idx: int) -> dict:
    image_id = self.unique_ids[idx]
    dicom_path = os.path.join(self.dicom_dir, f"{image_id}.dicom")
    
    # Load the DICOM image
    dicom = pydicom.dcmread(dicom_path)
    image = dicom.pixel_array
    if "PhotometricInterpretation" in dicom:
      if dicom.PhotometricInterpretation == "MONOCHROME1":
        image = np.amax(image) - image

    # to convert 16 bit image into 8 bit
    slope = dicom.RescaleSlope if "RescaleSlope" in dicom else 1.0
    intercept = dicom.RescaleIntercept if "RescaleIntercept" in dicom else 0.0
    image = (image.astype(np.float32) * slope) + intercept
    image = (image - image.min()) / (image.max() - image.min()) * 255
    image = np.stack([image, image, image])
    image = image.transpose(1, 2, 0).astype(np.uint8) / 255  # betwee [0, 1]
    
    image: np.ndarray = cv2.resize(image, (TRAIN_IMAGE_WIDTH, TRAIN_IMAGE_HEIGHT)).transpose(2, 0, 1) # out shape: (C, H, W)
    # Filter rows for the current image_id
    rows = self.df[self.df['image_id'] == image_id]
    
    targets = []
    orig_width, orig_height = dicom.pixel_array.shape
    for _, row in rows.iterrows():
      labels = row['class_id']
      if labels == 0:
        xmin = 0.0
        ymin = 0.0
        xmax = 1.0
        ymax = 1.0
        pass
      else:
        # Denormalize bounding box coordinates to final image size
        xmin: float = (row.get('x_min', 0) / orig_width) * TRAIN_IMAGE_WIDTH
        ymin: float = (row.get('y_min', 0) / orig_height) * TRAIN_IMAGE_HEIGHT
        xmax: float = (row.get('x_max', orig_width) / orig_width) * TRAIN_IMAGE_WIDTH
        ymax: float = (row.get('y_max', orig_height) / orig_height) * TRAIN_IMAGE_HEIGHT
      
      targets.append({
        "labels": labels,
        "bbox_coords": (xmin, ymin, xmax, ymax)
      })

    return {
      "images": torch.tensor(image),
      "targets": targets
    }

dataset = LocalizationVinDrDS(CSV_FILEPATH, DICOM_PATH)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True, prefetch_factor=170, drop_last=False)


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
assert (SHARD_SIZE % batch_size) == 0, f'SHARD_SIZE should be perfectly divisible by batch_size, but got {SHARD_SIZE} % {batch_size} = {SHARD_SIZE % batch_size}'

# TODO: use dynamic key names? they are repeated quite a lot
curr_shard = {'images': [], 'targets': []}
for batch in tqdm(dataloader, total=len(dataloader), desc="Saving shards"):
  curr_shard['images'].append(batch['images'].numpy())
  curr_shard['targets'].extend(batch['targets'])

  if len(curr_shard['images']) >= SHARD_SIZE//batch_size:
    shard_path = os.path.join(SHARD_DIR, f"shard_{shard_count:04d}.gz")
    final_shard = {
      'images': np.concatenate(curr_shard['images'], axis=0),
      'targets': curr_shard['targets'],
    }
    save_shard(final_shard, shard_path)
    shard_count += 1
    curr_shard = {'images': [], 'targets': []}

if len(curr_shard['images']) == SHARD_SIZE//batch_size:
  shard_path = os.path.join(SHARD_DIR, f"shard_{shard_count:04d}.gz")
  final_shard = {
    'images': np.concatenate(curr_shard['images'], axis=0),
    'targets': curr_shard['targets'],
  }
  save_shard(final_shard, shard_path)
