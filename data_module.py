from collections import defaultdict
import random

import numpy as np
import pandas as pd
import torch
import nibabel as nib
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, Sampler
from lightning import LightningDataModule

from utils import *


# Lookup table for converting segmentation labels to simplified tissue labels (used for normalization)
tissue_lut = {0: 0, 2: 1, 3: 2, 4: 3, 5: 3, 7: 1, 8: 2, 10: 2, 11: 2, 12: 2, 13: 2, 14: 3, 15: 3, 16: 4, 17: 2, 18: 2,
              24: 3, 26: 2, 28: 4, 41: 1, 42: 2, 43: 3, 44: 3, 46: 1, 47: 2, 49: 2, 50: 2, 51: 2, 52: 2, 53: 2, 54: 2,
              58: 2, 60: 4, 1001: 2, 1002: 2, 1003: 2, 1005: 2, 1006: 2, 1007: 2, 1008: 2, 1009: 2, 1010: 2, 1011: 2,
              1012: 2, 1013: 2, 1014: 2, 1015: 2, 1016: 2, 1017: 2, 1018: 2, 1019: 2, 1020: 2, 1021: 2, 1022: 2,
              1023: 2, 1024: 2, 1025: 2, 1026: 2, 1027: 2, 1028: 2, 1029: 2, 1030: 2, 1031: 2, 1032: 2, 1033: 2,
              1034: 2, 1035: 2, 2001: 2, 2002: 2, 2003: 2, 2005: 2, 2006: 2, 2007: 2, 2008: 2, 2009: 2, 2010: 2,
              2011: 2, 2012: 2, 2013: 2, 2014: 2, 2015: 2, 2016: 2, 2017: 2, 2018: 2, 2019: 2, 2020: 2, 2021: 2,
              2022: 2, 2023: 2, 2024: 2, 2025: 2, 2026: 2, 2027: 2, 2028: 2, 2029: 2, 2030: 2, 2031: 2, 2032: 2,
              2033: 2, 2034: 2, 2035: 2} 

def collate_func(batch):
    """
    Custom collate function for longitudinal batches.
    Handles batches of data with two images (longitudinal pairs) and their corresponding ages.

    Args:
        batch (list): List of tuples ((data1, data2), flag), where data1 and data2 are (image, age, participant_id).

    Returns:
        tuple: (images1, images2, ages1, ages2, flags, participant_ids)
    """
    images1, images2, ages1, ages2, flags, participant_ids = [], [], [], [], [], []

    for (data1, data2), flag in batch:
        image1, age1, pid1 = data1
        image2, age2, pid2 = data2

        images1.append(image1)
        images2.append(image2)
        ages1.append(age1)
        ages2.append(age2)
        flags.append(flag)
        participant_ids.append((pid1, pid2))  # Store both participant IDs

    images1 = torch.stack(images1)
    images2 = torch.stack(images2)
    ages1 = torch.tensor(ages1, dtype=torch.float64)
    ages2 = torch.tensor(ages2, dtype=torch.float64)
    
    return images1, images2, ages1, ages2, flags, participant_ids  # Return IDs
    
def collate_func_crossSec(batch):
    """
    Custom collate function for cross-sectional batches.
    Handles batches of data with single images and their corresponding ages.

    Args:
        batch (list): List of tuples (image, age, [participant_id]).

    Returns:
        tuple: (data, age, [sub_ids])
    """
    data = torch.stack([item[0] for item in batch])
    age = torch.stack([torch.tensor(item[1]) for item in batch])

    # Add singleton dimensions to the target tensor
    age = age.unsqueeze(1).unsqueeze(2).unsqueeze(3)

    # Check if the third element exists in the batch items
    if len(batch[0]) > 2:
        sub_ids = []
        for _, _, sub_id in batch:
            sub_ids.append(sub_id)
        return data, age, sub_ids
    else:
        return data, age

class LongitudinalBatchSampler(Sampler):
    """
    Custom batch sampler for longitudinal data.
    Groups data into pairs based on subject and session, and bins pairs by age difference.

    Args:
        data_source (Dataset): Dataset object.
        batch_size (int): Batch size (must be even).
    """
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        assert batch_size % 2 == 0, "Batch size should be even for pairing"

        # Filter dataset based on the split
        self.filtered_data = self.data_source.data[self.data_source.data['split'] == self.data_source.split].reset_index()

        # Organize sessions by subject
        self.subject_sessions = defaultdict(list)
        for idx, row in self.filtered_data.iterrows():
            pid = row['participant_id']
            self.subject_sessions[pid].append((idx, row['session'], row['age']))

        # Sort sessions by age for each subject
        for pid in self.subject_sessions:
            self.subject_sessions[pid].sort(key=lambda x: x[2])

        # Create longitudinal pairs and bin by age difference
        self.binned_pairs = {
            'short': [],
            'medium': [],
            'long': [],
        }

        self.cross_sectional_pairs = []
        self.single_subject_indices = []

        for pid, sessions in self.subject_sessions.items():
            indices = [x[0] for x in sessions]
            ages = [x[2] for x in sessions]

            if len(indices) == 1:
                self.single_subject_indices.append(indices[0])
            else:
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        age_diff = abs(ages[j] - ages[i])
                        pair = ([indices[i], indices[j]], True)

                        if age_diff <= 3:
                            self.binned_pairs['short'].append(pair)
                        elif age_diff < 7:
                            self.binned_pairs['medium'].append(pair)
                        else:
                            self.binned_pairs['long'].append(pair)

        # Create cross-sectional pairs from unpaired individuals
        random.shuffle(self.single_subject_indices)
        for i in range(0, len(self.single_subject_indices) - 1, 2):
            self.cross_sectional_pairs.append(([self.single_subject_indices[i], self.single_subject_indices[i + 1]], False))

        self.binned_pairs['cross'] = self.cross_sectional_pairs

        # Save bin names and gather all pairs
        self.bin_names = list(self.binned_pairs.keys())
        self.all_pairs = (
            self.binned_pairs['short']
            + self.binned_pairs['medium']
            + self.binned_pairs['long']
            + self.binned_pairs['cross']
        )

    def __iter__(self):
        """
        Yields batches of pairs, shuffled each epoch.
        """
        random.shuffle(self.all_pairs)          # fresh shuffle each epoch
        for i in range(0, len(self.all_pairs), self.batch_size):
            yield self.all_pairs[i : i + self.batch_size]

    def __len__(self):
        """
        Returns the number of batches per epoch.
        """
        return len(self.all_pairs) // self.batch_size

class BrainAgeDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for brain age prediction.
    Handles train/val/test splits and batch sampling.

    Args:
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.
        test_dataset (Dataset): Test dataset.
        batch_size (int): Batch size.
        num_workers (int): Number of workers for DataLoader.
        sampler (str): Sampler type ('LongitudinalSampler' or other).
        collate_function (callable): Collate function for DataLoader.
    """
    def __init__(self, 
                 train_dataset, 
                 val_dataset, 
                 test_dataset, 
                 batch_size, 
                 num_workers, 
                 sampler, 
                 collate_function = collate_func):
        
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_function = collate_function
        self.sampler = sampler

    def inspect_first_batch(self):
        """
        Prints the shapes of the first batch from the training dataloader.
        """
        first_batch = next(iter(self.train_dataloader()))
        input_data, target_data = first_batch
        print("Input data shape: ", input_data.shape)
        print("Target data shape: ", target_data.shape)

    def train_dataloader(self):
        """
        Returns the training DataLoader.
        Uses LongitudinalBatchSampler if specified, otherwise standard batching.
        """
        if self.sampler == 'LongitudinalSampler':
            sampler = LongitudinalBatchSampler(data_source=self.train_dataset, batch_size=self.batch_size)
            return DataLoader(self.train_dataset, batch_sampler=sampler, num_workers=self.num_workers, collate_fn=self.collate_function)
        else:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate_function)
    
    def val_dataloader(self):
        """
        Returns the validation DataLoader.
        Uses LongitudinalBatchSampler if specified, otherwise standard batching.
        """
        if self.sampler == 'LongitudinalSampler':
            sampler = LongitudinalBatchSampler(data_source=self.val_dataset, batch_size=self.batch_size)
            return DataLoader(self.train_dataset, batch_sampler=sampler, num_workers=self.num_workers, collate_fn=self.collate_function)
        else:
            return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_function)
    
    def test_dataloader(self):
        """
        Returns the test DataLoader.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_function)
            
class BrainAgingDataset(Dataset):
    """
    Dataset for brain aging prediction.
    Loads images, ages, and optionally segmentation masks and IDs.

    Args:
        csv_file (str): Path to the CSV file.
        split (str): Dataset split to use ('train', 'val', or 'test').
        image_type (str): Type of image to use ('T1w' or 'GM_VBM').
        wm_norm (bool): Whether to load the WM segmentation for normalization.
        skull_strip (bool): Whether to apply skull-stripping using segmentation.
        return_id (bool): Whether to return participant ID.
        return_seg (bool): Whether to return segmentation mask.
        bin_size (int): Bin size for age binning.
        transform (callable): Transform function for preprocessing.
        data_augm (bool): Whether to apply data augmentation.
    """
    def __init__(self, 
                 csv_file, 
                 split, 
                 image_type='T1w', 
                 return_id=False, 
                 return_seg=False, 
                 bin_size=5, 
                 transform=None, 
                 data_augm=False, 
                 wm_norm=False, 
                 skull_strip=False):
        self.data = pd.read_csv(csv_file)
        
        self.split = split
        self.image_type = image_type

        # Select image column based on type
        if image_type == 'T1w':
            self.image_col = 'T1aff_path'
        elif image_type == 'GM_VBM':
            self.image_col = 'VBM_GM_path'
        else:
            raise ValueError(f"Invalid image type '{image_type}'.")

        # Filter the dataframe based on the split
        self.data = self.data[self.data['split'] == split]

        self.transform = transform
        self.bin_size = bin_size
        self.data_augm = data_augm
        self.return_id = return_id
        self.image_type = image_type
        self.wm_norm = wm_norm
        self.skull_strip = skull_strip
        self.return_seg = return_seg
        
    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Retrieves the image and corresponding age (and optionally ID/segmentation) at the given index.

        Args:
            idx (int or tuple): Index or tuple of indices for longitudinal pairs.

        Returns:
            tuple: Data for one or two timepoints, depending on input.
        """

        if isinstance(idx, tuple):
            indices, flag = idx 
            idx1, idx2 = indices 
            row1 = self.data.iloc[idx1]
            row2 = self.data.iloc[idx2]
            data1 = self._load_data(row1)
            data2 = self._load_data(row2)
            return (data1, data2), flag
        else:
            row = self.data.iloc[idx]
            return self._load_data(row)
        
    def _load_data(self, row):
        """
        Loads image and optional segmentation for a single row.

        Args:
            row (pd.Series): Row of the dataframe.

        Returns:
            tuple: Image tensor, age, and optionally segmentation and participant ID.
        """
        image_path = row[self.image_col]
        image = nib.load(image_path)
        image = np.array(image.dataobj)
        image = np.expand_dims(image, axis=0)

        # Load the WM segmentation if specified
        seg = None
        if self.wm_norm:
            seg_path = row['synthseg_path']
            seg = nib.load(seg_path)
            seg = np.array(seg.dataobj).astype(int)
            segm_tissues = convert_labelmap(seg, tissue_lut)
            seg = np.where(segm_tissues == 1, 1, 0)
            seg = np.expand_dims(seg, axis=0) # Add channel dimension

        # Apply skull-stripping if specified
        if self.skull_strip:
            skull_seg_path = row['synthseg_path']
            skull_seg = nib.load(skull_seg_path)
            skull_seg = np.array(skull_seg.dataobj).astype(int)
            image = image * (skull_seg > 0)

        # Apply preprocessing/augmentation
        if self.transform is not None:
            augm = self.data_augm
            image = self.transform(image, seg, augm, self.image_type)

        age = row['age']
        participant_id = row['participant_id']
        
        # Return requested data
        if self.return_id:
            if self.return_seg:
                seg_path = row['synthseg_path']
                seg = nib.load(seg_path)
                seg = np.array(seg.dataobj)
                seg = crop_center(seg, (160, 192, 160))
                return image, seg, age, participant_id
            else:
                return image, age, participant_id
        else:
            if self.return_seg:
                seg_path = row['synthseg_path']
                seg = nib.load(seg_path)
                seg = np.array(seg.dataobj)
                seg = crop_center(seg, (160, 192, 160))
                return image, seg, age
            else:
                return image, age

class crossSecBrainAgingDataset(Dataset):
    """
    Dataset for cross-sectional brain aging prediction.
    Loads images, ages, and optionally segmentation masks, IDs, and session info.

    Args:
        csv_file (str): Path to the CSV file.
        split (str): Dataset split to use ('train', 'val', or 'test').
        image_type (str): Type of image to use ('T1w' or 'GM_VBM').
        wm_norm (bool): Whether to load the WM segmentation for normalization.
        skull_strip (bool): Whether to apply skull-stripping using segmentation.
        return_id (bool): Whether to return participant ID.
        return_seg (bool): Whether to return segmentation mask.
        return_session (bool): Whether to return session info.
        bin_size (int): Bin size for age binning.
        transform (callable): Transform function for preprocessing.
        data_augm (bool): Whether to apply data augmentation.
    """
    def __init__(self, 
                 csv_file, 
                 split, 
                 image_type='T1w', 
                 return_session=False, 
                 return_id=False, 
                 return_seg=False, 
                 bin_size=5, 
                 transform=None, 
                 data_augm=False, 
                 wm_norm=False, 
                 skull_strip=False):
        self.data = pd.read_csv(csv_file)
        self.split = split
        self.image_type = image_type

        # Select image column based on type
        if image_type == 'T1w':
            self.image_col = 'T1aff_path'
        elif image_type == 'GM_VBM':
            self.image_col = 'VBM_GM_path'
        else:
            raise ValueError(f"Invalid image type '{image_type}'.")

        # Filter the dataframe based on the split
        self.data = self.data[self.data['split'] == split]

        self.transform = transform
        self.bin_size = bin_size
        self.data_augm = data_augm
        self.return_id = return_id
        self.image_type = image_type
        self.wm_norm = wm_norm
        self.skull_strip = skull_strip
        self.return_seg = return_seg
        self.return_session = return_session
        
    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Retrieves the image and corresponding age (and optionally ID/segmentation/session) at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: Data for the requested sample.
        """
        row = self.data.iloc[idx]
        image_path = row[self.image_col]
        image = nib.load(image_path)
        image = np.array(image.dataobj)
        image = np.expand_dims(image, axis=0)

        # Load the WM segmentation if specified
        seg = None
        if self.wm_norm:
            seg_path = row['synthseg_path']
            seg = nib.load(seg_path)
            seg = np.array(seg.dataobj).astype(int)
            segm_tissues = convert_labelmap(seg, tissue_lut)
            seg = np.where(segm_tissues == 1, 1, 0)
            seg = np.expand_dims(seg, axis=0) 

        # Apply skull-stripping if specified
        if self.skull_strip:
            skull_seg_path = row['synthseg_path']
            skull_seg = nib.load(skull_seg_path)
            skull_seg = np.array(skull_seg.dataobj).astype(int)
            image = image * (skull_seg > 0)

        # Apply preprocessing/augmentation
        if self.transform is not None:
            augm = self.data_augm
            image = self.transform(image, seg, augm, self.image_type)

        age = row['age']
        participant_id = row['participant_id']
        
        # Return requested data
        if self.return_id:
            if self.return_seg:
                if self.return_session:
                    seg_path = row['synthseg_path']
                    seg = nib.load(seg_path)
                    seg = np.array(seg.dataobj)
                    seg = crop_center(seg, (160, 192, 160))
                    return image, seg, age, participant_id, session
                else:
                    seg_path = row['synthseg_path']
                    seg = nib.load(seg_path)
                    seg = np.array(seg.dataobj)
                    seg = crop_center(seg, (160, 192, 160))
                    return image, seg, age, participant_id
            else:
                if self.return_session:
                    session = row['session']
                    return image, age, participant_id, session
                else:
                    return image, age, participant_id
        else:
            if self.return_seg:
                if self.return_session:
                    seg_path = row['synthseg_path']
                    seg = nib.load(seg_path)
                    seg = np.array(seg.dataobj)
                    seg = crop_center(seg, (160, 192, 160))
                    return image, seg, age, session
                else:
                    seg_path = row['synthseg_path']
                    seg = nib.load(seg_path)
                    seg = np.array(seg.dataobj)
                    seg = crop_center(seg, (160, 192, 160))
                    return image, seg, age
            else:
                if self.return_session:
                    return image, age, session
                else:
                    return image, age