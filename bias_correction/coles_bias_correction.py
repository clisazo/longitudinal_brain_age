import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_absolute_error

experiment = '' # Name of the experiment folder

# Load the validation and test datasets
val_df = pd.read_csv(f'/path/to/validation/results/val_results.csv')
test_df = pd.read_csv(f'/path/to/test/results/test_results.csv')

# Fit Huber Regressor on validation data to estimate the parameters alpha and beta
huber = HuberRegressor().fit(val_df[['groundtruth_age']], val_df['predicted_age'])
alpha, beta = huber.coef_[0], huber.intercept_

print(f"Estimated parameters: alpha = {alpha:.4f}, beta = {beta:.4f}")

# Calculate PAD for the test dataset before correction
test_df['PAD'] = test_df['predicted_age'] - test_df['groundtruth_age']

# Apply bias correction to the predicted age
test_df['Y_hat_c'] = (test_df['predicted_age'] - beta) / alpha

# Recalculate the corrected PAD
test_df['PADc'] = test_df['Y_hat_c'] - test_df['groundtruth_age']

# Save the updated test dataframe to a new CSV file
output_path = f'/path/to/bias_corrected/output/test_results_corrected.csv'
test_df.to_csv(output_path, index=False)

# Calculate MAE before correction
mae_before = mean_absolute_error(test_df['groundtruth_age'], test_df['predicted_age'])

# Calculate MAE after correction
mae_after = mean_absolute_error(test_df['groundtruth_age'], test_df['Y_hat_c'])

# Compute absolute errors
absolute_errors_before = np.abs(test_df['groundtruth_age'] - test_df['predicted_age'])
absolute_errors_after = np.abs(test_df['groundtruth_age'] - test_df['Y_hat_c'])

# Compute standard deviation of absolute errors
std_before = np.std(absolute_errors_before, ddof=1)  # Using ddof=1 for sample standard deviation
std_after = np.std(absolute_errors_after, ddof=1)

print(f"MAE before correction: {mae_before:.4f}, Std Dev: {std_before:.4f}")
print(f"MAE after correction: {mae_after:.4f}, Std Dev: {std_after:.4f}")