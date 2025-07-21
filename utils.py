import random

import numpy as np
import torch
import torch.nn as nn
import torchio as tio
import SimpleITK as sitk
from scipy.stats import norm

def crop_center(data, out_sp):
    """
    Crops the center part of a 3D or 4D volume to the specified output shape.

    Args:
        data (np.ndarray): Input volume (3D or 4D).
        out_sp (tuple): Desired output shape (z, y, x).

    Returns:
        np.ndarray: Cropped volume.
    """
    in_sp = data.shape
    nd = np.ndim(data)
    x_crop = int((in_sp[-1] - out_sp[-1]) / 2)
    y_crop = int((in_sp[-2] - out_sp[-2]) / 2)
    z_crop = int((in_sp[-3] - out_sp[-3]) / 2)
    if nd == 3:
        data_crop = data[x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop]
    elif nd == 4:
        data_crop = data[:, x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop]
    else:
        raise ValueError(f"Wrong dimension! dim={nd}.")
    return data_crop

def num2vect(x, bin_range, bin_step, sigma):
    """
    Converts a number or array of numbers to a probability vector representation over bins.

    Args:
        x (float or np.ndarray): Input value(s).
        bin_range (tuple): (start, end) of bin range.
        bin_step (int): Step size for bins.
        sigma (float): Standard deviation for Gaussian smoothing.
            0: hard label (returns bin index)
            >0: soft label (returns probability vector)
            <0: error

    Returns:
        tuple: (vector or indices, bin_centers)
    """
    bin_start = bin_range[0]
    bin_end = bin_range[1]
    bin_length = bin_end - bin_start + 1
    if not bin_length % bin_step == 0:
        print("bin's range should be divisible by bin_step!")
        return -1
    bin_number = int(bin_length / bin_step)
    bin_centers = bin_start + float(bin_step) / 2 + bin_step * np.arange(bin_number)

    if sigma == 0:
        x = np.array(x)
        i = np.floor((x - bin_start) / bin_step)
        i = i.astype(int)
        return i, bin_centers
    elif sigma > 0:
        if np.isscalar(x):
            v = np.zeros((bin_number,))
            for i in range(bin_number):
                x1 = bin_centers[i] - float(bin_step) / 2
                x2 = bin_centers[i] + float(bin_step) / 2
                cdfs = norm.cdf([x1, x2], loc=x, scale=sigma)
                v[i] = cdfs[1] - cdfs[0]
            return v, bin_centers
        else:
            v = np.zeros((len(x), bin_number))
            for j in range(len(x)):
                for i in range(bin_number):
                    x1 = bin_centers[i] - float(bin_step) / 2
                    x2 = bin_centers[i] + float(bin_step) / 2
                    cdfs = norm.cdf([x1, x2], loc=x[j], scale=sigma)
                    v[j, i] = cdfs[1] - cdfs[0]
            return v, bin_centers

def my_KLDivLoss(x, y):
    """
    Computes Kullback-Leibler Divergence loss averaged over the batch.

    Args:
        x (torch.Tensor): Log-probabilities (output of log_softmax).
        y (torch.Tensor): Target probability distributions.

    Returns:
        torch.Tensor: KL divergence loss.
    """
    loss_func = nn.KLDivLoss(reduction='sum')
    y = y + 1e-16  # Prevent log(0)
    n = y.shape[0]
    loss = loss_func(x, y) / n
    return loss

def standardize(img):
    """
    Scales image intensities to [0, 255].

    Args:
        img (np.ndarray): Input image.

    Returns:
        np.ndarray: Standardized image.
    """
    return (img - np.min(img)) * 255 / (np.max(img) - np.min(img))

def convert_labels(lut):
    """
    Converts a lookup table to a label array.

    Args:
        lut (dict): Lookup table mapping input labels to output labels.

    Returns:
        np.ndarray: Array of output labels.
    """
    labels = np.zeros((np.max(list(lut.keys())) + 1,))
    for in_lab, out_lab in lut.items():
        labels[in_lab] = out_lab
    return labels

def convert_labelmap(labelmap, lut):
    """
    Converts a labelmap using a lookup table.

    Args:
        labelmap (np.ndarray): Input labelmap.
        lut (dict): Lookup table.

    Returns:
        np.ndarray: Converted labelmap.
    """
    labels = convert_labels(lut)
    L = labels[labelmap]
    return L

def one_hot_encoding(target, num_classes=None, categories=None):
    """
    Converts target array to one-hot encoding.

    Args:
        target (np.ndarray): Target array.
        num_classes (int, optional): Number of classes.
        categories (dict or list, optional): Category mapping.

    Returns:
        np.ndarray: One-hot encoded array.
    """
    if categories is None and num_classes is None:
        categories = np.sort(np.unique(target))
        num_classes = len(categories)
    elif categories is not None:
        if isinstance(categories, list) or isinstance(categories, np.ndarray):
            categories = {cls: it_cls for it_cls, cls in enumerate(categories)}
        num_classes = len(np.unique(np.array(list(categories.values()))))
    else:
        categories = {cls: cls for cls in np.arange(num_classes)}

    labels = np.zeros((num_classes,) + target.shape, dtype='int')
    for cls, it_cls in categories.items():
        idx_class = np.where(target == cls)
        idx = (it_cls,) + idx_class
        labels[idx] = 1
    return labels

def binarize_image(image, threshold):
    """
    Binarizes an image based on a threshold.

    Args:
        image (np.ndarray): Input image.
        threshold (float): Threshold value.

    Returns:
        np.ndarray: Binary mask.
    """
    binary_mask = np.zeros_like(image, dtype=np.uint8)
    binary_mask[image >= threshold] = 1
    return binary_mask

def apply_augmentation(image, image_type):
    """
    Applies random augmentation (flip, shift, rotation, intensity transforms) to an image.

    Args:
        image (np.ndarray): Input image.
        image_type (str): Type of image (e.g., 'T1w').

    Returns:
        np.ndarray: Augmented image.
    """
    # Flip along axes
    flip_probabilities = [0, 0.5, 0]
    shift_amounts = [2, 2, 2]
    rotation_angle = 5

    for axis, prob in enumerate(flip_probabilities):
        if np.random.uniform() < prob:
            image = np.flip(image, axis=axis)

    # Shift along axes
    for i in range(3):
        shift = np.random.randint(-shift_amounts[i], shift_amounts[i] + 1)
        image = np.roll(image, shift, axis=i)

    # Rotate image
    angle = np.random.uniform(-rotation_angle, rotation_angle)
    image = rotate_image(image, angle)

    # Intensity transforms for T1w images
    if image_type == 'T1w':
        intensity_transforms = {
            tio.RandomBiasField(): 0.5,
            tio.RandomBlur(): 0.5
        }
        transform = tio.Compose([
            tio.OneOf(intensity_transforms, p=0.5),
            tio.RescaleIntensity(out_min_max=(0, 1)),
        ])
        image = transform(image)

    return image

def rotate_image(image, angle):
    """
    Rotates each 2D slice of a 3D image by a given angle.

    Args:
        image (np.ndarray): 3D image (depth, height, width).
        angle (float): Rotation angle in degrees.

    Returns:
        np.ndarray: Rotated image.
    """
    rotated_image = np.stack([rotate_single_slice(slice, angle) for slice in image])
    return rotated_image

def rotate_single_slice(image_slice, angle):
    """
    Rotates a single 2D image slice.

    Args:
        image_slice (np.ndarray): 2D image slice.
        angle (float): Rotation angle in degrees.

    Returns:
        np.ndarray: Rotated slice.
    """
    from scipy.ndimage import rotate
    rotated_slice = rotate(image_slice, angle, reshape=False, mode='nearest')
    return rotated_slice

def shift_distribution(dist, bin_centers, shift_value):
    """
    Shifts a probability distribution along bin centers by a given value.

    Args:
        dist (torch.Tensor): Distribution tensor.
        bin_centers (torch.Tensor): Bin centers.
        shift_value (float): Value to shift.

    Returns:
        torch.Tensor: Shifted distribution.
    """
    device = dist.device
    bin_centers = bin_centers.to(device)
    shifted_dist = torch.zeros_like(dist)
    for j in range(len(bin_centers)):
        shifted_bin = bin_centers[j] - shift_value
        closest_idx = (torch.abs(bin_centers - shifted_bin)).argmin()
        shifted_dist[closest_idx] += dist[j]
    return shifted_dist

def set_seed(seed: int):
    """
    Sets random seed for reproducibility.

    Args:
        seed (int): Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train_transform(image, seg=None, augm=False, image_type=None):
    """
    Preprocesses and augments training images.

    Args:
        image (np.ndarray): Input image (channels, ...).
        seg (np.ndarray, optional): Segmentation mask.
        augm (bool): Whether to apply augmentation.
        image_type (str, optional): Type of image.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    normalized_images = []
    for i, channel_data in enumerate(image):
        if i == 0:
            if seg is not None:
                white_matter_mean = np.mean(channel_data[seg[0] == 1])
                normalized_channel = channel_data / white_matter_mean
            else:
                img_max = np.max(channel_data)
                normalized_channel = channel_data / img_max
                normalized_channel = normalized_channel / np.mean(normalized_channel)
        else:
            normalized_channel = channel_data
        cropped_channel = crop_center(normalized_channel, (160, 192, 160))
        normalized_images.append(cropped_channel)
    data_array = np.array(normalized_images, dtype=np.float32)
    if augm and random.random() < 0.3: 
        data_array = apply_augmentation(data_array, image_type)
    data_tensor = torch.tensor(data_array, dtype=torch.float32)
    return data_tensor

def val_transform(image, seg=None, augm=False, image_type=None):
    """
    Preprocesses validation images (no augmentation).

    Args:
        image (np.ndarray): Input image (channels, ...).
        seg (np.ndarray, optional): Segmentation mask.
        augm (bool): Unused, kept for API consistency.
        image_type (str, optional): Unused, kept for API consistency.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    normalized_images = []
    for i, channel_data in enumerate(image):
        if i == 0:
            if seg is not None:
                white_matter_mean = np.mean(channel_data[seg[0] == 1])
                normalized_channel = channel_data / white_matter_mean
            else:
                img_max = np.max(channel_data)
                normalized_channel = channel_data / img_max
                normalized_channel = normalized_channel / np.mean(normalized_channel)
        else:
            normalized_channel = channel_data
        cropped_channel = crop_center(normalized_channel, (160, 192, 160))
        normalized_images.append(cropped_channel)
    data_array = np.array(normalized_images, dtype=np.float32)
    data_tensor = torch.tensor(data_array, dtype=torch.float32)
    return data_tensor

def test_transform(image, seg=None, augm=False, image_type=None):
    """
    Preprocesses test images (no augmentation).

    Args:
        image (np.ndarray): Input image (channels, ...).
        seg (np.ndarray, optional): Segmentation mask.
        augm (bool): Unused, kept for API consistency.
        image_type (str, optional): Unused, kept for API consistency.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    normalized_images = []
    for i, channel_data in enumerate(image):
        if i == 0:
            if seg is not None:
                white_matter_mean = np.mean(channel_data[seg[0] == 1])
                normalized_channel = channel_data / white_matter_mean
            else:
                img_max = np.max(channel_data)
                normalized_channel = channel_data / img_max
                normalized_channel = normalized_channel / np.mean(normalized_channel)
        else:
            normalized_channel = channel_data
        cropped_channel = crop_center(normalized_channel, (160, 192, 160))
        normalized_images.append(cropped_channel)
    data_array = np.array(normalized_images, dtype=np.float32)
    data_tensor = torch.tensor(data_array, dtype=torch.float32)
    return data_tensor