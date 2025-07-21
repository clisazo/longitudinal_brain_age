CONFIG = {
    "seed": 42, # random seed for reproducibility
    "gpus": '0', # GPU to use
    "architecture": 'SFCN_NoAvgPooling', # 'DenseNet' or 'SFCN' or 'SFCN_NoAvgPooling' or 'ColesNet'
    "experiment_name": '', # Name of the experiment to load the model from
    "ckpt_path": '', # Path to the checkpoint file (if None, will use the latest checkpoint)
    "image_type": "T1w", # Options: 'GM_VBM', 'T1w'
    "in_channels": 1, # Number of input channels
    "bin_range": [35, 100], # Range of age bins
    "bin_step": 1, # Step for age bins
    "sigma": 1, # Sigma for Gaussian smoothing of age probabilities
    "output_dim": 66, # Number of output bins
    "data_augm": False, # Whether to use data augmentation
    "wm_norm": True, # Whether to normalize by white matter intensity
    "skull_strip": True, # Whether to skull strip images
    "longitudinal_training": True, # Whether to load a model trained with longitudinal-consistency constraint
    "experiment_root": "/path/to/experiments/root/directory", # Root directory for experiments
    "general_data_csv_path": "/path/to/dataset.csv", # Path to the CSV containing dataset information
    "ADNI_AD_MCI_path": '/path/to/ADNI/ADNI_AD_MCI.csv', # Path to the CSV file containing ADNI AD/MCI dataset information
    "MIRIAD_csv_path": '/path/to/MIRIAD/MIRIAD_dataset.csv', # Path to the CSV file containing external (MIRIAD) dataset information
}