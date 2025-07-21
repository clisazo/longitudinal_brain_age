import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random
import torch
from sklearn.metrics import r2_score
from tqdm import tqdm

# Add the main folder to PYTHONPATH for imports
sys.path.append(os.path.abspath('..'))

from config_inference import CONFIG
from data_module import *
from model_module import *
from utils import test_transform, set_seed

def load_model():
    """
    Loads the trained model from checkpoint based on configuration.

    Returns:
        torch.nn.Module: Loaded model ready for inference.
    """
    ckpts_path = os.path.join(CONFIG['experiment_root'], CONFIG['experiment_name'], 'Model_checkpoints')
    ckpt_path = os.path.join(ckpts_path, sorted(os.listdir(ckpts_path))[-1])
    ckpt_path = CONFIG['ckpt_path'] if CONFIG['ckpt_path'] else ckpt_path
    print(f"Loading model from {ckpt_path}")

    # Instantiate model architecture
    if CONFIG['architecture'] == 'SFCN_NoAvgPooling':
        model = SFCN_NoAvgPooling(input_channels=CONFIG['in_channels'], output_dim=CONFIG['output_dim'])
    elif CONFIG['architecture'] == 'SFCN':
        model = SFCN(input_channels=CONFIG['in_channels'], output_dim=CONFIG['output_dim'])
    elif CONFIG['architecture'] == 'DenseNet':
        model = DenseNetWithLogSoftmax(spatial_dims=3, in_channels=CONFIG['in_channels'], out_channels=CONFIG['output_dim'])
    elif CONFIG['architecture'] == 'ColesNet':
        model = ColesNet(input_channels=CONFIG['in_channels'], output_dim=CONFIG['output_dim'])
    else:
        raise ValueError("Invalid architecture. Choose between 'SFCN', 'ColesNet' and 'DenseNet'.")

    # Load checkpoint into LightningModule
    if CONFIG['longitudinal_training']:
        model = longitudinalRegressionModel.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            model=model,
            bin_range=CONFIG['bin_range'],
            bin_step=CONFIG['bin_step'],
            sigma=CONFIG['sigma'],
            loss='KLDiv',
            lr=1E-2,
            data_module=None,
            optim='SGD',
            lr_sch=True,
            lr_sch_type='StepLR',
            weight_decay=0,
            max_epochs=500
        )
    else:
        model = regressionModel.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            model=model,
            bin_range=CONFIG['bin_range'],
            bin_step=CONFIG['bin_step'],
            sigma=CONFIG['sigma'],
            loss='KLDiv',
            lr=1E-2,
            data_module=None,
            optim='SGD',
            lr_sch=True,
            lr_sch_type='StepLR',
            weight_decay=0,
            max_epochs=500
        )

    return model.to(device)

def run_inference(model, dataset_csv, split):
    """
    Runs inference on a cross-sectional dataset and collects predictions and metadata.

    Args:
        model (torch.nn.Module): Trained model.
        dataset_csv (str): Path to CSV file with dataset info.
        split (str): Dataset split to use ('val', 'test', etc.).

    Returns:
        tuple: Lists of predictions, labels, sexes, IDs, sessions, and results.
    """
    data_info_df = pd.read_csv(dataset_csv)
    id_to_sex = dict(zip(data_info_df['participant_id'], data_info_df['sex']))

    dataset = crossSecBrainAgingDataset(
        csv_file=dataset_csv,
        split=split,
        image_type=CONFIG['image_type'],
        return_id=True,
        return_session=True,
        train_resample=CONFIG['train_resample'],
        wm_norm=CONFIG['wm_norm'],
        skull_strip=CONFIG['skull_strip'],
        transform=test_transform
    )

    print(f"Starting inference on {split} dataset. Number of images: {len(dataset)}")

    results_list = []
    all_predictions, all_labels, all_sexes, all_ids, all_sessions = [], [], [], [], []

    model.eval()
    with torch.no_grad():
        for idx in tqdm(range(len(dataset))):
            input_img, label, participant_id, session = dataset[idx]
            input_img = input_img.unsqueeze(0).to(device)
            label = float(label)

            # Convert label to soft label (probability distribution)
            y, bc = num2vect(torch.tensor(label).view(1, -1).cpu(),
                             bin_range=CONFIG['bin_range'], 
                             bin_step=CONFIG['bin_step'], 
                             sigma=CONFIG['sigma'])
            y_hat = model(input_img)[0].cpu().numpy().squeeze()
            pred = np.exp(y_hat) @ bc 

            # Store results
            all_predictions.append(pred)
            all_labels.append(label)
            all_ids.append(participant_id)
            all_sexes.append(id_to_sex[participant_id])
            all_sessions.append(session)
            results_list.append([participant_id, label, pred, id_to_sex[participant_id], session])

    return all_predictions, all_labels, all_sexes, all_ids, all_sessions, results_list

def run_inference_miriad(model, dataset_csv, split, session):
    """
    Runs inference on the MIRIAD dataset for a specific session.

    Args:
        model (torch.nn.Module): Trained model.
        dataset_csv (str): Path to CSV file with dataset info.
        split (str): Dataset split to use.
        session (str): Session identifier.

    Returns:
        tuple: Lists of predictions, labels, sexes, IDs, sessions, and results.
    """
    data_info_df = pd.read_csv(dataset_csv)
    id_to_sex = dict(zip(data_info_df['participant_id'], data_info_df['sex']))

    dataset = crossSecBrainAgingDataset(
        csv_file=dataset_csv,
        split=split,
        image_type=CONFIG['image_type'],
        return_id=True,
        return_session=False,
        train_resample=CONFIG['train_resample'],
        wm_norm=CONFIG['wm_norm'],
        skull_strip=CONFIG['skull_strip'],
        transform=test_transform
    )

    print(f"Starting inference on {split} dataset. Number of images: {len(dataset)}")

    results_list = []
    all_predictions, all_labels, all_sexes, all_ids, all_sessions = [], [], [], [], []

    model.eval()
    with torch.no_grad():
        for idx in tqdm(range(len(dataset))):
            input_img, label, participant_id = dataset[idx]
            input_img = input_img.unsqueeze(0).to(device)
            label = float(label)

            # Convert label to soft label (probability distribution)
            y, bc = num2vect(torch.tensor(label).view(1, -1).cpu(),
                             bin_range=CONFIG['bin_range'], 
                             bin_step=CONFIG['bin_step'], 
                             sigma=CONFIG['sigma'])
            y_hat = model(input_img)[0].cpu().numpy().squeeze()
            pred = np.exp(y_hat) @ bc 

            # Store results
            all_predictions.append(pred)
            all_labels.append(label)
            all_ids.append(participant_id)
            all_sexes.append(id_to_sex[participant_id])
            all_sessions.append(session)
            results_list.append([participant_id, label, pred, id_to_sex[participant_id], session])

    return all_predictions, all_labels, all_sexes, all_ids, all_sessions, results_list

def compute_metrics(all_labels, all_predictions):
    """
    Computes evaluation metrics for predictions.

    Args:
        all_labels (list): Ground truth ages.
        all_predictions (list): Predicted ages.

    Returns:
        tuple: Correlation coefficient, MAE, std of MAE, R-squared, MAE per decade.
    """
    all_labels, all_predictions = np.array(all_labels), np.array(all_predictions)

    valid_indices = ~np.isnan(all_labels) & ~np.isnan(all_predictions)
    all_labels = all_labels[valid_indices]
    all_predictions = all_predictions[valid_indices]
    
    correlation_coefficient = np.corrcoef(all_labels, all_predictions)[0, 1]
    mae = np.mean(np.abs(all_predictions - all_labels))
    std = np.std(np.abs(all_predictions - all_labels))
    r_squared = r2_score(all_labels, all_predictions)
    
    # Compute MAE per decade
    age_ranges_decades = [(i, i + 9) for i in range(10, 85, 10)]
    mae_decades = {
        decade: np.mean(np.abs(all_predictions[np.where((all_labels >= decade[0]) & (all_labels <= decade[1]))] - 
                               all_labels[np.where((all_labels >= decade[0]) & (all_labels <= decade[1]))]))
        for decade in age_ranges_decades
    }

    return correlation_coefficient, mae, std, r_squared, mae_decades

# Function to save results to CSV and text file
def save_metrics(experiment_name, results_list, correlation_coefficient, mae, std, r_squared, mae_decades, save_csv_path, save_metrics_path, ad_mci=False):
    """
    Saves inference results and metrics to CSV and text files.

    Args:
        experiment_name (str): Name of the experiment.
        results_list (list): List of results per sample.
        correlation_coefficient (float): Correlation coefficient.
        mae (float): Mean Absolute Error.
        std (float): Standard deviation of MAE.
        r_squared (float): R-squared value.
        mae_decades (dict): MAE per decade.
        save_csv_path (str): Filename for CSV.
        save_metrics_path (str): Filename for metrics text file.
        ad_mci (bool): If True, save to AD_MCI subfolder.
    """
    if ad_mci:
        result_path = os.path.join(CONFIG['experiment_root'], f"{experiment_name}/results_csv/AD_MCI")
    else:
        result_path = os.path.join(CONFIG['experiment_root'], f"{experiment_name}/results_csv")
    os.makedirs(result_path, exist_ok=True)

    # Save CSV
    results_df = pd.DataFrame(results_list, columns=['participant_id', 'groundtruth_age', 'predicted_age', 'sex', 'session'])
    results_df.to_csv(os.path.join(result_path, save_csv_path), index=False)
    
    # Save metrics to text file
    metrics_path = os.path.join(result_path, save_metrics_path)
    with open(metrics_path, "w") as f:
        f.write(f"Correlation coefficient: {correlation_coefficient:.4f}\n")
        f.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")
        f.write(f"Std of MAE: {std:.4f}\n")
        f.write(f"R-squared: {r_squared:.4f}\n")
        for decade, mae_decade in mae_decades.items():
            f.write(f"MAE for ages {decade[0]}-{decade[1]}: {mae_decade:.4f}\n")

    print(f"Results saved to {result_path}")

# Function to plot results
def plot_results(all_labels, all_predictions, experiment_name, plot_name, ad_mci=False):
    """
    Plots predicted vs. real ages and saves the figure.

    Args:
        all_labels (list): Ground truth ages.
        all_predictions (list): Predicted ages.
        experiment_name (str): Name of the experiment.
        plot_name (str): Filename for the plot.
        ad_mci (bool): If True, save to AD_MCI subfolder.
    """
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    plt.figure(figsize=(8, 6))
    plt.scatter(all_labels, all_predictions, color='blue', alpha=0.5, label='Predictions')
    plt.plot([all_labels.min(), all_labels.max()], [all_labels.min(), all_labels.max()], 
             color='red', linestyle='--', linewidth=2, label='Perfect correlation')
    plt.title("Correlation between real and predicted ages")
    plt.xlabel("Real age")
    plt.ylabel("Predicted age")
    plt.grid(True)
    plt.legend()
    
    if ad_mci:
        save_path = os.path.join(CONFIG['experiment_root'], f"{experiment_name}/plots/AD_MCI")
    else:
        save_path = os.path.join(CONFIG['experiment_root'], f"{experiment_name}/plots")
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{plot_name}.png"))


# Set random seed for reproducibility
set_seed(CONFIG["seed"])

# Set matrix multiplication precision for torch
torch.set_float32_matmul_precision('high')

# Set CUDA devices
os.environ['CUDA_VISIBLE_DEVICES'] = CONFIG['gpus']
gpusToUse = [0]
device = torch.device(f"cuda:{gpusToUse[0]}" if torch.cuda.is_available() else "cpu")

#############################################################################
#################### INFERENCE ON THE VAL SET ###############################
#############################################################################

# Run inference and save results for validation set
model = load_model()
all_predictions, all_labels, all_sexes, all_ids, all_sessions, results_list = run_inference(model=model, dataset_csv=CONFIG['general_data_csv_path'], split='val')

# Compute and save metrics
correlation_coefficient, mae, std, r_squared, mae_decades = compute_metrics(all_labels, all_predictions)
save_metrics(CONFIG['experiment_name'], results_list, correlation_coefficient, mae, std, r_squared, mae_decades, 'val_results_with_sex.csv', 'val_metrics.txt')

# Plot results
plot_results(all_labels, all_predictions, CONFIG['experiment_name'], plot_name='val_correlation_plot')


#############################################################################
#################### INFERENCE ON THE TEST SET ##############################
#############################################################################

# Run inference and save results for test set
model = load_model()
all_predictions, all_labels, all_sexes, all_ids, all_sessions, results_list = run_inference(model=model, dataset_csv=CONFIG['general_data_csv_path'], split='test')

# Compute and save metrics
correlation_coefficient, mae, std, r_squared, mae_decades = compute_metrics(all_labels, all_predictions)
save_metrics(CONFIG['experiment_name'], results_list, correlation_coefficient, mae, std, r_squared, mae_decades, 'test_results_with_sex.csv', 'test_metrics.txt')

# Plot results
plot_results(all_labels, all_predictions, CONFIG['experiment_name'], plot_name='test_correlation_plot')


#############################################################################
#################### INFERENCE ON ADNI  DATASET #############################
#############################################################################

# Run inference for each session in ADNI dataset (healthy controls and AD/MCI)
sessions = ['000', '003', '006', '012', '018', '024', '030', '036', '042', '048', '054', '060', '066', '072', '078', '084', '090', '096', '102', '108', '120', '132', '144', '156', '180']

for session in sessions:
    print(f"Starting inference on ADNI dataset for session {session}")
    model = load_model()

    ################################ AD AND MCI ################################    
    all_predictions, all_labels, all_sexes, all_ids, all_sessions, results_list = run_inference(model=model, dataset_csv=CONFIG['ADNI_AD_MCI_path'], split=f'test_m{session}')

    # Compute and save metrics
    correlation_coefficient, mae, std, r_squared, mae_decades = compute_metrics(all_labels, all_predictions)
    save_metrics(CONFIG['experiment_name'], results_list, correlation_coefficient, mae, std, r_squared, mae_decades, f'ADNI_{session}_test_results_with_sex.csv', f'ADNI_{session}_metrics.txt', ad_mci=True)

    # Plot results
    plot_results(all_labels, all_predictions, CONFIG['experiment_name'], f'ADNI_session_{session}_correlation_plot',ad_mci=True)


#############################################################################
################### INFERENCE ON MIRIAD  DATASET ############################
#############################################################################

# Run inference for each session in MIRIAD dataset
numbers = ['01', '02', '03', '04', '05', '06', '07', '09']

for number in numbers:
    print(f"Starting inference on MIRIAD dataset for session {number}")
    model = load_model()
    all_predictions, all_labels, all_sexes, all_ids, all_sessions, results_list = run_inference_miriad(model=model, dataset_csv=CONFIG['MIRIAD_csv_path'], split=f'test_{number}', session=number)

    # Compute and save metrics
    correlation_coefficient, mae, std, r_squared, mae_decades = compute_metrics(all_labels, all_predictions)
    save_metrics(CONFIG['experiment_name'], results_list, correlation_coefficient, mae, std, r_squared, mae_decades, f'MIRIAD_ses-{number}_results.csv', f'MIRIAD_ses-{number}_metrics.txt')

    # Plot results
    plot_results(all_labels, all_predictions, CONFIG['experiment_name'], f'MIRIAD_session_{number}_correlation_plot')

