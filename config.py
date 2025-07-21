CONFIG = {
    "seed": 42, # random seed for reproducibility
    "gpus": '3', # GPU to use
    "architecture_name": 'DenseNet', # Options: SFCN_NoAvgPooling, SFCN, DenseNet, ColesNet
    "dropout": False, # Whether to use dropout in the model
    "batch_size": 2, # Batch size for training
    "num_workers": 4, # Number of workers for data loading
    "max_epochs": 500, # Maximum number of epochs for training
    "patience": 50, # Patience to stop training and save best model
    "learning_rate": 1E-4, # Learning rate for the optimizer
    "lr_scheduler": "StepLR", # Learning rate scheduler type (Options: StepLR, OneCycleLR, ReduceLROnPlateau)
    "weight_decay": 0, # Weight decay for optimizer
    "loss": "KLDiv",  # Loss function. Options: KLDiv, CrossEntropy
    "monitor_metric": "val_mae", # Metric to monitor for early stopping
    "optimizer": "Adam", # Options: SGD, Adam
    "image_type": "T1w",  # Options: 'GM_VBM', 'T1w'
    "in_channels": 1, # Number of input channels
    "bin_range": [35, 100], # Range of age bins
    "bin_step": 1, # Step for age bins
    "sigma": 1, # Sigma for Gaussian smoothing of age probabilities
    "output_dim": 66,  # Number of output neurons (bins)
    "data_augm": False, # Whether to use data augmentation
    "wm_norm": True, # Whether to normalize by white matter intensity
    "skull_strip": True, # whether to skull strip images
    "return_id": True, # Whether to return the subject id in the dataset (good for debugging)
    "longitudinal_training": True, # Whether to use longitudinal-consistency constraint for training
    "sampler": 'LongitudinalSampler',  # Way of sampling data (options: None (when not using longitudinal constraint), LongitudinalSampler (when using longitudinal constraint))
    "lambda_consistency": 1, # Weight of the consistency loss term
    "resume_from_ckpt": None, # Path to checkpoint to resume training from (None to start from scratch)
    "experiment_number": "999", # Experiment number for logging
    "experiment_root": "/path/to/experiments/root/directory", # Root directory for experiments
    "data_csv_path": "/path/to/dataset.csv", # Path to the CSV file containing dataset information
    "fast_dev_run": False # Run only one batch for debugging if set to True
}