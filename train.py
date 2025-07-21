from datetime import datetime
import os
import random

import json
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import torch

from config import CONFIG 
from data_module import BrainAgeDataModule, BrainAgingDataset, collate_func, collate_func_crossSec
from model_module import SFCN, SFCN_NoAvgPooling, DenseNetWithLogSoftmax, ColesNet, longitudinalRegressionModel, regressionModel
from utils import *

def create_experiment_dir():
    """
    Creates a directory for the current experiment, including subfolders for logs, checkpoints, and results.

    Returns:
        str: Path to the experiment directory.
    """
    folder_name = (
        f"{CONFIG['experiment_number']}-{datetime.now().strftime('%Y-%m-%d')}-"
        f"{CONFIG['architecture_name']}-Loss-{CONFIG['loss']}-{CONFIG['optimizer']}-"
        f"output_dim-{CONFIG['output_dim']}-LR-{CONFIG['learning_rate']}-monitor-{CONFIG['monitor_metric']}"
    )
    experiment_path = os.path.join(CONFIG['experiment_root'], folder_name)
    os.makedirs(experiment_path, exist_ok=True)
    for subfolder in ['lightning_logs', 'Model_checkpoints', 'Results']:
        os.makedirs(os.path.join(experiment_path, subfolder), exist_ok=True)
    return experiment_path

def create_datasets():
    """
    Creates train, validation, and test datasets using configuration parameters.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    common_args = {
        "csv_file": CONFIG['data_csv_path'],
        "image_type": CONFIG['image_type'],
        "wm_norm": CONFIG['wm_norm'],
        "skull_strip": CONFIG['skull_strip'],
        "return_id": CONFIG['return_id']
    }
    return (
        BrainAgingDataset(split='train', transform=train_transform, **common_args),
        BrainAgingDataset(split='val', transform=val_transform, **common_args),
        BrainAgingDataset(split='test', transform=val_transform, **common_args)
    )


# Set random seed for reproducibility
set_seed(CONFIG["seed"])

# Set matrix multiplication precision for torch
torch.set_float32_matmul_precision('high')

# Set CUDA devices
os.environ['CUDA_VISIBLE_DEVICES'] = CONFIG['gpus']
gpusToUse = [0]
device = torch.device(f"cuda:{gpusToUse[0]}" if torch.cuda.is_available() else "cpu")

# Create experiment directory
experiment_path = create_experiment_dir()

# Create datasets
train_dataset, val_dataset, test_dataset = create_datasets()
print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# Select collate function based on training mode
if CONFIG['longitudinal_training']:
    collate_function = collate_func
else:
    collate_function = collate_func_crossSec

# Initialize DataModule
brainAgeDM = BrainAgeDataModule(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
    batch_size=CONFIG['batch_size'],
    num_workers=CONFIG['num_workers'],
    sampler=CONFIG['sampler'],
    collate_function=collate_function
)

# Model setup based on architecture name
if CONFIG['architecture_name'] == 'SFCN_NoAvgPooling':
    model = SFCN_NoAvgPooling(input_channels=CONFIG['in_channels'], output_dim=CONFIG['output_dim'], dropout=CONFIG['dropout'])
elif CONFIG['architecture_name'] == 'SFCN':
    model = SFCN(input_channels=CONFIG['in_channels'], output_dim=CONFIG['output_dim'], dropout=CONFIG['dropout'])
elif CONFIG['architecture_name'] == 'DenseNet':
    model = DenseNetWithLogSoftmax(spatial_dims=3, in_channels=CONFIG['in_channels'], out_channels=CONFIG['output_dim'], dropout_prob=0.0 if not CONFIG['dropout'] else 0.2)
elif CONFIG['architecture_name'] == 'ColesNet':
    model = ColesNet(input_channels=CONFIG['in_channels'], output_dim=CONFIG['output_dim'])

# Wrap model in appropriate LightningModule
if CONFIG['longitudinal_training']:
    model = longitudinalRegressionModel(
        model=model,
        architecture_name=CONFIG['architecture_name'],
        bin_range=CONFIG['bin_range'],
        bin_step=CONFIG['bin_step'],
        sigma=CONFIG['sigma'],
        loss=CONFIG['loss'],
        lr=CONFIG['learning_rate'],
        data_module=brainAgeDM,
        optim=CONFIG['optimizer'],
        lr_sch=True,
        lr_sch_type=CONFIG['lr_scheduler'],
        weight_decay=CONFIG['weight_decay'],
        max_epochs=CONFIG['max_epochs'],
        lambda_consistency_max=CONFIG['lambda_consistency'],
        image_type=CONFIG['image_type']
    )
else:
    model = regressionModel(
        model=model,
        architecture_name=CONFIG['architecture_name'],
        bin_range=CONFIG['bin_range'],
        bin_step=CONFIG['bin_step'],
        sigma=CONFIG['sigma'],
        loss=CONFIG['loss'],
        lr=CONFIG['learning_rate'],
        data_module=brainAgeDM,
        optim=CONFIG['optimizer'],
        lr_sch=True,
        lr_sch_type=CONFIG['lr_scheduler'],
        weight_decay=CONFIG['weight_decay'],
        max_epochs=CONFIG['max_epochs'],
        image_type=CONFIG['image_type']
    )
model.to(device)

# Define callbacks for early stopping and model checkpointing
callbacks = [
    EarlyStopping(monitor=CONFIG['monitor_metric'], patience=CONFIG['patience'], mode='min', verbose=True),
    ModelCheckpoint(
        dirpath=os.path.join(experiment_path, 'Model_checkpoints'),
        filename='model-{epoch:02d}-{val_mae:.2f}',
        monitor=CONFIG['monitor_metric'],
        mode='min',
        save_top_k=1,
        verbose=False
    )
]

# Set up TensorBoard logger
logger = TensorBoardLogger(os.path.join(experiment_path, 'lightning_logs'))

# Initialize PyTorch Lightning Trainer
trainer = Trainer(
    max_epochs=CONFIG['max_epochs'],
    accelerator="gpu",
    devices=gpusToUse,
    callbacks=callbacks,
    deterministic=False,
    fast_dev_run=CONFIG['fast_dev_run'],
    enable_model_summary=True,
    logger=logger
)

# Train the model, optionally resuming from checkpoint
if CONFIG['resume_from_ckpt']:
    trainer.fit(model, brainAgeDM, ckpt_path=CONFIG['resume_from_ckpt'])
else:
    trainer.fit(model, brainAgeDM)