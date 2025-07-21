from lightning import LightningModule
from monai.networks.nets import DenseNet121
import numpy as np
import torch
import torchmetrics
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, OneCycleLR

from utils import *
        
class SFCN(nn.Module):
    """"
    Simple Fully Convolutional Network (SFCN) for 3D data. As introduced in:
    
     @article{Peng_Gong_Beckmann_Vedaldi_Smith_2021, 
     title={Accurate brain age prediction with lightweight deep neural networks}, 
     volume={68}, 
     ISSN={13618415}, 
     url={https://linkinghub.elsevier.com/retrieve/pii/S1361841520302358}, 
     DOI={10.1016/j.media.2020.101871}, 
     journal={Medical Image Analysis}, 
     author={Peng, Han and Gong, Weikang and Beckmann, Christian F. and Vedaldi, Andrea and Smith, Stephen M.}, 
     year={2021}, 
     month=feb, 
     pages={101871}}

    Args:
        input_channels (int): Number of input channels.
        channel_number (list): List of output channels for each convolutional layer.
        output_dim (int): Number of output classes.
        dropout (bool): Whether to include dropout in the classifier.
        activation (bool): Whether to apply log_softmax activation.

    """
    def __init__(self, input_channels = 1, channel_number=[32, 64, 128, 256, 256, 64], output_dim = 40, dropout=True, activation=True):
        super().__init__()
        self.activation = activation
        n_layer = len(channel_number)
        self.feature_extractor = nn.Sequential()

        for i in range(n_layer):
            if i == 0:
                in_channel = input_channels
            else:
                in_channel = channel_number[i-1]
            out_channel = channel_number[i]
            if i < n_layer-1:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=True,
                                                                  kernel_size=3,
                                                                  padding=1))
            else:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=False,
                                                                  kernel_size=1,
                                                                  padding=0))
        self.classifier = nn.Sequential()
        avg_shape = [5, 6, 5]
        self.classifier.add_module('average_pool', nn.AvgPool3d(avg_shape))
        if dropout is True:
            self.classifier.add_module('dropout', nn.Dropout(0.5))
        i = n_layer
        in_channel = channel_number[-1]
        out_channel = output_dim
        self.classifier.add_module('conv_%d' % i,
                                   nn.Conv3d(in_channel, out_channel, padding=0, kernel_size=1))

    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
        if maxpool is True:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )
        return layer

    def forward(self, x):
        out = list()
        x_f = self.feature_extractor(x)
        x = self.classifier(x_f)
        if self.activation:
            x = F.log_softmax(x, dim=1)
        out.append(x)
        return out
    
class SFCN_NoAvgPooling(nn.Module):
    """"
    Simple Fully Convolutional Network (SFCN) for 3D data, adapted to replace 
    average pooling with a fully connected layer for classification.
    
    This adaptation removes the global average pooling layer present in the 
    original SFCN architecture and flattens the feature extractor's output
    before passing it through a linear layer for final classification.

    Args:
        input_channels (int): Number of input channels. Defaults to 1.
        channel_number (list): List of output channels for each convolutional layer.
        output_dim (int): Number of output classes. Defaults to 66.
        dropout (bool): If True, a dropout layer is applied before the final fully connected layer. Defaults to True.
        activation (bool): If True, log_softmax activation is applied to the output for multi-class classification. Defaults to True.
    """
    def __init__(self, input_channels=1, channel_number=[32, 64, 128, 256, 256, 64], output_dim=66, dropout=True, activation=True):
        super().__init__()
        self.activation = activation
        self.feature_extractor = nn.Sequential()

        n_layer = len(channel_number)
        for i in range(n_layer):
            in_channel = input_channels if i == 0 else channel_number[i - 1]
            out_channel = channel_number[i]
            maxpool = True if i < n_layer - 1 else False
            kernel_size = 3 if maxpool else 1
            padding = 1 if maxpool else 0
            self.feature_extractor.add_module(
                f'conv_{i}',
                self.conv_layer(in_channel, out_channel, maxpool, kernel_size, padding)
            )

        self.dropout = nn.Dropout(0.5) if dropout else nn.Identity()

        self.fc = nn.Linear(64 * 5 * 6 * 5, output_dim)

    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
        layers = [
            nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm3d(out_channel),
            nn.ReLU()
        ]
        if maxpool:
            layers.append(nn.MaxPool3d(2, stride=maxpool_stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)  
        x = self.dropout(x)
        x = self.fc(x)
        if self.activation:
            x = F.log_softmax(x, dim=1)
        return [x]


class DenseNetWithLogSoftmax(DenseNet121):
    """
    A DenseNet121 model with an added log_softmax activation in the forward pass.

    This class inherits from MONAI's DenseNet121 and modifies its forward method
    to apply a log_softmax function to the output, which is used for
    the multi-class classification task to obtain log-probabilities.

    Args:
        spatial_dims (int): Number of spatial dimensions of the input image.
        in_channels (int): Number of input channels to the network.
        out_channels (int): Number of output channels (i.e. number of classes).
        dropout_prob (float, optional): Dropout probability for the DenseNet's final classification layer.
                                        Defaults to 0.0 (no dropout).
        **kwargs: Additional keyword arguments to be passed to the DenseNet121 parent class constructor.
    """
    def __init__(self, spatial_dims, in_channels, out_channels, dropout_prob=0.0, **kwargs):
        # Call the constructor of the parent class (DenseNet121)
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_prob=dropout_prob,
            **kwargs
        )

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): The input tensor to the model.

        Returns:
            torch.Tensor: The output tensor after passing through DenseNet121 and applying log_softmax.
        """
        
        x = super().forward(x)
        return F.log_softmax(x, dim=1)

class ColesNet(nn.Module):
    """Implementation of Cole et al.'s model for brain age estimation."""
    def __init__(self, input_channels=1, output_dim=40):
        super().__init__()

        # 5 blocks of:
        # - 3x3x3 conv layer stride = 1 + ReLU
        # - 3x3x3 conv + batch normalization layer + ReLU
        # - 2x2x2 max pooling layer stride = 2
        # One fully connected layer
        # The number of features is set to 8 in the 1st block and doubled after each max pooling layer
        self.conv1 = nn.Conv3d(input_channels, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(8, 8, kernel_size=3, stride=1, padding=1)
        self.bnorm1 = nn.BatchNorm3d(8, affine=True)

        self.conv3 = nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bnorm2 = nn.BatchNorm3d(16, affine=True)

        self.conv5 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bnorm3 = nn.BatchNorm3d(32, affine=True)

        self.conv7 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bnorm4 = nn.BatchNorm3d(64, affine=True)

        self.conv9 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bnorm5 = nn.BatchNorm3d(128, affine=True)

        self.fc1 = nn.Linear(19200, output_dim)

    def forward(self, x):
        # 1st block
        x = F.relu(self.conv1(x))
        x = F.relu(self.bnorm1(self.conv2(x)))
        x = F.max_pool3d(x, kernel_size=2, stride=2)

        # 2nd block
        x = F.relu(self.conv3(x))
        x = F.relu(self.bnorm2(self.conv4(x)))
        x = F.max_pool3d(x, kernel_size=2, stride=2)

        # 3rd block
        x = F.relu(self.conv5(x))
        x = F.relu(self.bnorm3(self.conv6(x)))
        x = F.max_pool3d(x, kernel_size=2, stride=2)

        # 4th block
        x = F.relu(self.conv7(x))
        x = F.relu(self.bnorm4(self.conv8(x)))
        x = F.max_pool3d(x, kernel_size=2, stride=2)

        # 5th block
        x = F.relu(self.conv9(x))
        x = F.relu(self.bnorm5(self.conv10(x)))
        x = F.max_pool3d(x, kernel_size=2, stride=2)

        # Fully connected layer
        x = x.view(-1, self.num_flat_features(x))
        # print("x shape", x.shape)
        x = self.fc1(x)
        x = F.log_softmax(x, dim=1)  # Apply log_softmax along the class dimension
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:] # All dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
class ColesNet(nn.Module):
    """
    Implementation of Cole et al.'s model for brain age estimation.

    Reference:
    @article{Cole_2017,
    title={Predicting brain age with deep learning from raw imaging data results in a reliable and heritable biomarker},
    volume={163},
    journal={NeuroImage},
    author={Cole, James H. and Poudel, Rudra P. K. and Tsagkrasoulis, Dimosthenis and Caan, Matthan W. A. and Steves, Claire and Spector, Tim D. and Montana, Giovanni},
    year={2017},
    month=dec,
    pages={115-124} }

    Args:
        input_channels (int, optional): Number of channels in the input image. Defaults to 1.
        output_dim (int, optional): Dimension of the output layer. Defaults to 66.
    """
    class ColeBlock(nn.Module):
        """
        Consists of two 3D convolutional layers, a batch normalization layer,
        ReLU activations, and a max pooling layer.
        """
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.bnorm = nn.BatchNorm3d(out_channels, affine=True)
            self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.bnorm(self.conv2(x)))
            x = self.maxpool(x)
            return x

    def __init__(self, input_channels=1, output_dim=66):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            ColesNet.ColeBlock(input_channels, 8),
            ColesNet.ColeBlock(8, 16),
            ColesNet.ColeBlock(16, 32),
            ColesNet.ColeBlock(32, 64),
            ColesNet.ColeBlock(64, 128)
        )

        self.fc1 = nn.Linear(19200, output_dim)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = F.log_softmax(x, dim=1)
        return x
    
    def num_flat_features(self, x):
        """
        Calculates the total number of features after flattening a tensor.
        """
        size = x.size()[1:] # All dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    
    
class regressionModel(LightningModule):
    """
    Pytorch Lightning module to train a brain age prediction model.

    Args:
        model (nn.Module): The neural network model to be trained.
        loss (str): The loss function to be used for training. (options are: 'KLDiv', 'CrossEntropy')
        lr (float): Learning rate for the optimizer.
        optim (str): Type of optimizer to use ('SGD' or 'Adam').
        architecture_name (str): Name of the architecture being used. Options are 'SFCN', 'SFCN_NoAvgPooling', 'DenseNetWithLogSoftmax', 'ColesNet'. Defaults to 'SFCN'.
        image_type (str): Type of input images (e.g., 'T1w').
        weight_decay (float): Weight decay for the optimizer. Defaults to 0.
        data_module: Data module containing training and validation data loaders.
        lr_sch (bool): Whether to use learning rate scheduling. Defaults to True.
        lr_sch_type (str): Type of learning rate scheduler to use. Options are 'ReduceLROnPlateau', 'StepLR', or 'OneCycleLR'. Defaults to 'StepLR'.
        max_epochs (int, optional): Maximum number of epochs for training. Defaults to 500.
        bin_range (list, optional): Range for binning ages. Defaults to [35, 100].
        bin_step (int, optional): Step size for binning ages. Defaults to 1.
        sigma (float, optional): Standard deviation for Gaussian smoothing in age binning. Defaults to 1.
    """
    def __init__(self, 
                 model, 
                 loss, 
                 lr, 
                 optim, 
                 architecture_name='SFCN', 
                 image_type = 'T1w', 
                 weight_decay=0, 
                 data_module = None, 
                 lr_sch = True, 
                 lr_sch_type='StepLR', 
                 max_epochs=500, 
                 bin_range = [35,100], 
                 bin_step=1, 
                 sigma = 1):
        
        super().__init__()
        self.model = model
        self.lr = lr
        self.lr_sch = lr_sch
        self.architecture_name = architecture_name
        self.mae = torchmetrics.MeanAbsoluteError()
        self.optim = optim
        self.weight_decay = weight_decay
        self.lr_sch_type = lr_sch_type
        self.max_epochs = max_epochs
        self.data_module = data_module
        self.bin_range = bin_range
        self.bin_step = bin_step
        self.sigma = sigma
        self.image_type = image_type

        # Select loss function
        if loss == 'KLDiv':
            self.loss = my_KLDivLoss
        elif loss == 'CrossEntropy':
            self.loss = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss}")
        
    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)
    
    def configure_optimizers(self):
        """Configures the optimizer and learning rate scheduler."""
        # Choose optimizer
        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        elif self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optim}")

        # Optionally add learning rate scheduler
        if self.lr_sch:
            if self.lr_sch_type == 'ReduceLROnPlateau':
                self.lr_scheduler = {
                    'scheduler': ReduceLROnPlateau(self.optimizer, mode='min', patience = 10, factor = 0.1, verbose=True),
                    'monitor': 'val_mae',
                    'interval': 'epoch',
                    'frequency': 1
                }
            elif self.lr_sch_type == 'StepLR':
                self.lr_scheduler = {
                    'scheduler': StepLR(self.optimizer, step_size=30, gamma=0.3, verbose=False),
                    'monitor': 'val_mae',
                    'interval': 'epoch',
                    'frequency': 1
                }
            elif self.lr_sch_type == 'OneCycleLR':
                max_lr = self.lr
                # total_steps = len(self.train_dataloader()) * self.max_epochs
                self.lr_scheduler = {
                    'scheduler': OneCycleLR(self.optimizer, max_lr=max_lr, steps_per_epoch=len(self.data_module.train_dataloader()), epochs=self.max_epochs),
                    'interval': 'step',
                    'frequency': 1
                }
            else:
                raise ValueError(f"Unsupported learning rate scheduler type: {self.lr_sch_type}. Must be either 'ReduceLROnPlateau', 'StepLR' or 'OneCycleLR'.")
            return{
                'optimizer': self.optimizer,
                'lr_scheduler': self.lr_scheduler
            }
        else:
            return self.optimizer
        
    def shared_step(self, batch, batch_idx, prefix):
        """
        Shared logic for training and validation steps.

        Args:
            batch: Input batch containing (x, y, participant_id).
            batch_idx: Batch index.
            prefix: 'train' or 'val' for logging.
        Returns:
            loss: Computed loss for the batch.
        """
        x, y, participant_id = batch
        batch_size = x.shape[0]
        y = y.float().squeeze(2).squeeze(2)
        label = y

        # Convert ages to soft labels (probability distributions)
        y_np, bc = num2vect(y.cpu().numpy(), self.bin_range, self.bin_step, self.sigma)
        y = torch.tensor(y_np, dtype=torch.float32).to(x.device)

        # Forward pass
        out = self.forward(x)
        if (self.architecture_name == 'SFCN' or self.architecture_name == 'SFCN_NoAvgPooling'):
            y_hat = out[0].squeeze()
        else:
            y_hat = out

        # Convert logits to probabilities and predict age
        prob = torch.exp(y_hat)
        pred = torch.matmul(prob, torch.tensor(bc, dtype=torch.float32, device=prob.device))
        label_squeezed = label.squeeze().to(pred.device)
        mae = torch.mean(torch.abs(pred - label_squeezed))
        
        # Compute loss
        loss = self.loss(y_hat, y)

        # Log metrics
        self.log(f'{prefix}_loss', loss)
        self.log(f"{prefix}_mae", mae, on_step=(prefix=='train'), on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        """Training step: calls shared_step with 'train' prefix."""
        return self.shared_step(batch, batch_idx, prefix='train')

    def validation_step(self, batch, batch_idx):
        """Validation step: calls shared_step with 'val' prefix."""
        return self.shared_step(batch, batch_idx, prefix='val')



class longitudinalRegressionModel(LightningModule):
    """
    Pytorch Lightning module for training a brain age prediction model with longitudinal-consistency constraint.
    
    Args:
        model (nn.Module): The neural network model to be trained.
        loss (str): The loss function to be used for training. (options are: 'KLDiv', 'CrossEntropy')
        lr (float): Learning rate for the optimizer.
        optim (str): Type of optimizer to use ('SGD' or 'Adam').
        architecture_name (str): Name of the architecture being used. Options are 'SFCN', 'SFCN_NoAvgPooling', 'DenseNetWithLogSoftmax', 'ColesNet'. Defaults to 'SFCN'.
        image_type (str): Type of input images (e.g., 'T1w').
        weight_decay (float): Weight decay for the optimizer. Defaults to 0.
        data_module: Data module containing training and validation data loaders.
        lr_sch (bool): Whether to use learning rate scheduling. Defaults to True.
        lr_sch_type (str): Type of learning rate scheduler to use. Options are 'ReduceLROnPlateau', 'StepLR', or 'OneCycleLR'. Defaults to 'StepLR'.
        max_epochs (int, optional): Maximum number of epochs for training. Defaults to 500.
        bin_range (list, optional): Range for binning ages. Defaults to [35, 100].
        bin_step (int, optional): Step size for binning ages. Defaults to 1.
        sigma (float, optional): Standard deviation for Gaussian smoothing in age binning. Defaults to 1.
        lambda_consistency_max (float, optional): Maximum value for the consistency loss weight. Defaults to 2.
    """
    def __init__(self, 
                 model, 
                 loss, 
                 lr, 
                 optim, 
                 architecture_name='SFCN', 
                 image_type='T1w', 
                 weight_decay=0, 
                 data_module = None, 
                 lr_sch = True, 
                 lr_sch_type='StepLR', 
                 max_epochs=500, 
                 bin_range = [35,100], 
                 bin_step=1, 
                 sigma = 1, 
                 lambda_consistency_max=2):
        
        super().__init__()
        self.model = model
        self.lr = lr
        self.lr_sch = lr_sch
        self.lr_sch_type = lr_sch_type
        self.optim = optim
        self.architecture_name = architecture_name
        self.image_type = image_type
        self.weight_decay = weight_decay
        self.data_module = data_module
        self.max_epochs = max_epochs
        self.bin_range = bin_range
        self.bin_step = bin_step
        self.sigma = sigma
        self.lambda_consistency = 0.0  # Initial value for consistency loss weight
        self.lambda_consistency_max = lambda_consistency_max
        self.lambda_consistency_warmup_epochs = 10 # Number of epochs in which the weight of the consistency loss is 0
        self.lambda_ramup_epochs = 10 # Number of epochs in which the weight of the consistency loss is linearly ramped up to its maximum value
        self.mae = torchmetrics.MeanAbsoluteError()

        # Select loss function
        if loss == 'KLDiv':
            self.loss = my_KLDivLoss
        elif loss == 'CrossEntropy':
            self.loss = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss}")
        
    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)
    
    def configure_optimizers(self):
        """Configures the optimizer and learning rate scheduler."""
        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        elif self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optim}")

        if self.lr_sch:
            if self.lr_sch_type == 'ReduceLROnPlateau':
                self.lr_scheduler = {
                    'scheduler': ReduceLROnPlateau(self.optimizer, mode='min', patience = 10, factor = 0.1, verbose=True),
                    'monitor': 'val_mae',
                    'interval': 'epoch',
                    'frequency': 1
                }
            elif self.lr_sch_type == 'StepLR':
                self.lr_scheduler = {
                    'scheduler': StepLR(self.optimizer, step_size=30, gamma=0.3, verbose=False),
                    'monitor': 'val_mae',
                    'interval': 'epoch',
                    'frequency': 1
                }
            elif self.lr_sch_type == 'OneCycleLR':
                max_lr = self.lr
                # total_steps = len(self.train_dataloader()) * self.max_epochs
                self.lr_scheduler = {
                    'scheduler': OneCycleLR(self.optimizer, max_lr=max_lr, steps_per_epoch=len(self.data_module.train_dataloader()), epochs=self.max_epochs),
                    'interval': 'step',
                    'frequency': 1
                }
            else:
                raise ValueError(f"Unsupported learning rate scheduler type: {self.lr_sch_type}. Must be either 'ReduceLROnPlateau', 'StepLR' or 'OneCycleLR'.")
            return{
                'optimizer': self.optimizer,
                'lr_scheduler': self.lr_scheduler
            }
        else:
            return self.optimizer
        
    def shared_step(self, batch, batch_idx, prefix):
        """
        Shared logic for training and validation steps.

        Args:
            batch: Input batch containing (images1, images2, ages1, ages2, flags, participant_ids).
            batch_idx: Batch index.
            prefix: 'train' or 'val' for logging.
        Returns:
            loss: Computed loss for the batch.
        """
        # Unpack the batch
        images1, images2, ages1, ages2, flags, participant_ids = batch
        batch_size = images1.shape[0]

        # Transform ages to soft labels (probability distributions)
        ages1_np, bc = num2vect(ages1.cpu().numpy(), self.bin_range, self.bin_step, self.sigma)
        ages2_np, _ = num2vect(ages2.cpu().numpy(), self.bin_range, self.bin_step, self.sigma)
        ages1_soft = torch.tensor(ages1_np, dtype=torch.float32).to(images1.device)
        ages2_soft = torch.tensor(ages2_np, dtype=torch.float32).to(images2.device)

        # Compute predictions for both sets of images
        out1 = self.forward(images1)
        out2 = self.forward(images2)

        # Extract logits depending on architecture
        if (self.architecture_name == 'SFCN' or self.architecture_name == 'SFCN_NoAvgPooling'):
            y_hat1 = out1[0].squeeze()
            y_hat2 = out2[0].squeeze()
        else:
            y_hat1 = out1
            y_hat2 = out2

        # Calculate probabilities
        prob1 = torch.exp(y_hat1)
        prob2 = torch.exp(y_hat2)

        # Calculate predicted ages from probabilities
        pred1 = torch.matmul(prob1, torch.tensor(bc, dtype=torch.float32, device=prob1.device))
        pred2 = torch.matmul(prob2, torch.tensor(bc, dtype=torch.float32, device=prob2.device))

        # Calculate Mean Absolute Error for each prediction
        mae1 = torch.mean(torch.abs(pred1 - ages1.to(pred1.device)))
        mae2 = torch.mean(torch.abs(pred2 - ages2.to(pred2.device)))
        mae = (mae1 + mae2) / 2

        # Calculate the primary loss
        primary_loss1 = self.loss(y_hat1, ages1_soft)
        primary_loss2 = self.loss(y_hat2, ages2_soft)
        primary_loss = (primary_loss1 + primary_loss2) / 2

        # Compute the longitudinal-consistency loss based on flags
        count = 0
        consistency_loss = 0
        eps = 1e-6  # To avoid division by zero
        for i in range(batch_size):
            if flags[i]:  # Same subject
                pred_difference = pred2[i] - pred1[i]
                gt_difference = ages2[i] - ages1[i]
                consistency_loss += torch.abs((pred_difference / (gt_difference + eps)) - 1)
                count += 1

        if count != 0:
            consistency_loss /= count
        else:
            consistency_loss = 0

        # Total loss: primary + weighted consistency
        loss = primary_loss + self.lambda_consistency * consistency_loss

        # Log metrics
        self.log(f"{prefix}_loss", loss)
        self.log(f"{prefix}_primary_loss", primary_loss)
        self.log(f"{prefix}_consistency_loss", consistency_loss)
        self.log(f"{prefix}_mae", mae, prog_bar=True, logger=True)
        if prefix == "train":
            self.log("lambda_consistency", self.lambda_consistency)

        return loss
    
    def training_step(self, batch, batch_idx):
        """Training step: calls shared_step with 'train' prefix."""
        return self.shared_step(batch, batch_idx, prefix='train')

    def validation_step(self, batch, batch_idx):
        """Validation step: calls shared_step with 'val' prefix."""
        return self.shared_step(batch, batch_idx, prefix='val')

    def on_train_epoch_start(self):
        """
        Adjusts lambda_consistency at the start of each epoch for warmup and ramp-up.
        """
        if self.current_epoch < self.lambda_consistency_warmup_epochs:
            self.lambda_consistency = 0.0
        elif self.current_epoch < self.lambda_consistency_warmup_epochs + self.lambda_ramup_epochs:
            # Linearly increase lambda_consistency from 0 to lambda_consistency_max
            progress = (self.current_epoch - self.lambda_consistency_warmup_epochs) / self.lambda_ramup_epochs
            self.lambda_consistency = self.lambda_consistency_max * progress
        else:
            self.lambda_consistency = self.lambda_consistency_max

    def on_validation_epoch_end(self):
        """
        Logs the mean validation MAE at the end of each validation epoch.
        """
        val_mae_values = self.trainer.callback_metrics.get('val_mae', [])
        mean_val_mae = torch.tensor(val_mae_values).mean().item()
        self.log('mean_val_mae', mean_val_mae, prog_bar=True, logger=True)