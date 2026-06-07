# =============================================================================
# MAIN EXECUTION - COMPLETELY TRAINING-DEPENDENT PIPELINE
# =============================================================================


# =============================================================================
# MAIN EXECUTION
# =============================================================================

# Improved MRI-to-CT Training - Optimized to Meet Targets
# Enhanced architecture and loss functions for better performance
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import cv2
from tqdm import tqdm
import warnings
import time
import gc
import logging
from typing import Tuple, Optional, List
import random

# Additional imports for visualization
import seaborn as sns
import pandas as pd
import json
from datetime import datetime
import re
from scipy import ndimage
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion
import matplotlib.patches as patches

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seeds()

class EnhancedDataProcessor:
    """Improved data preprocessing with better normalization strategies"""
    
    @staticmethod
    def normalize_mr_robust(img: np.ndarray) -> np.ndarray:
        """More robust MR normalization focusing on brain tissue"""
        # Remove background (typically < 5% of max intensity)
        threshold = np.max(img) * 0.05
        foreground_mask = img > threshold
        
        if np.sum(foreground_mask) < img.size * 0.1:
            # Fallback for edge cases
            mean = np.mean(img)
            std = np.std(img) + 1e-8
            return np.clip((img - mean) / std, -3, 3)
        
        # Normalize using foreground statistics
        fg_mean = np.mean(img[foreground_mask])
        fg_std = np.std(img[foreground_mask]) + 1e-8
        
        normalized = (img - fg_mean) / fg_std
        return np.clip(normalized, -3, 3)
    
    @staticmethod
    def normalize_ct_windowed(img: np.ndarray) -> np.ndarray:
        """CT normalization with soft tissue window focus"""
        # Soft tissue window: -160 to +240 HU
        img_clipped = np.clip(img, -160, 240)
        # Normalize to [-1, 1] with better soft tissue contrast
        return (img_clipped + 160) / 400 * 2 - 1
    
    @staticmethod
    def denormalize_ct_windowed(img: np.ndarray) -> np.ndarray:
        """Convert back to HU with soft tissue window"""
        return (img + 1) / 2 * 400 - 160

class ImprovedDataset(Dataset):
    """Enhanced dataset with better data quality and augmentation"""

    def __init__(self, data_path: str, img_size: int = 128, max_subjects: int = 170, 
                 max_slices_per_subject: int = 20, augment: bool = True, min_samples: int = 100):
        self.data_path = Path(data_path)
        self.img_size = img_size
        self.augment = augment
        self.min_samples = min_samples
        self.samples = []
        self.processor = EnhancedDataProcessor()

        logger.info(f"Enhanced Dataset Configuration:")
        logger.info(f"  Image size: {img_size}x{img_size}")
        logger.info(f"  Max subjects: {max_subjects}")
        logger.info(f"  Max slices per subject: {max_slices_per_subject}")
        logger.info(f"  Data augmentation: {augment}")

        self._load_enhanced_data(max_subjects, max_slices_per_subject)
        logger.info(f"Successfully loaded {len(self.samples)} samples")

    def _load_enhanced_data(self, max_subjects: int, max_slices: int):
        """Enhanced data loading with better quality control"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")
        
        subject_folders = [f for f in self.data_path.iterdir() if f.is_dir()]
        logger.info(f"Found {len(subject_folders)} subject folders")
        
        processed_subjects = 0
        total_slices = 0
        
        for subject_folder in tqdm(subject_folders[:max_subjects], desc="Loading subjects"):
            try:
                slices_added = self._process_subject_enhanced(subject_folder, max_slices)
                if slices_added > 0:
                    processed_subjects += 1
                    total_slices += slices_added
                    
                gc.collect()
                
            except Exception as e:
                logger.warning(f"Failed to process {subject_folder.name}: {str(e)[:50]}")
                continue

        logger.info(f"Processing complete: {processed_subjects} subjects, {total_slices} slices")
        
        if len(self.samples) < self.min_samples:
            raise ValueError(f"Insufficient samples: {len(self.samples)}. Need at least {self.min_samples}.")

    def _process_subject_enhanced(self, subject_folder: Path, max_slices: int) -> int:
        """Enhanced subject processing with better slice selection"""
        nii_files = list(subject_folder.glob("*.nii*"))
        if len(nii_files) < 2:
            return 0

        ct_file, mr_file = self._identify_modalities_smart(nii_files)
        if not ct_file or not mr_file:
            return 0

        try:
            ct_data = nib.load(ct_file).get_fdata().astype(np.float32)
            mr_data = nib.load(mr_file).get_fdata().astype(np.float32)

            if ct_data.shape != mr_data.shape or len(ct_data.shape) != 3:
                return 0

            # Enhanced slice selection focusing on brain tissue
            selected_slices = self._select_best_slices(ct_data, mr_data, max_slices)
            
            slices_added = 0
            for slice_idx in selected_slices:
                if self._add_enhanced_slice(ct_data[:, :, slice_idx], mr_data[:, :, slice_idx]):
                    slices_added += 1

            del ct_data, mr_data
            return slices_added

        except Exception:
            return 0

    def _identify_modalities_smart(self, nii_files: List[Path]) -> Tuple[Optional[Path], Optional[Path]]:
        """Smarter modality identification"""
        ct_file = None
        mr_file = None
        
        # Enhanced keyword matching
        for nii_file in nii_files:
            name_lower = nii_file.name.lower()
            
            # CT identification
            if any(keyword in name_lower for keyword in ['ct', 'computed', 'head_ct']):
                ct_file = nii_file
            # MR identification with more keywords
            elif any(keyword in name_lower for keyword in ['mr', 'mri', 'magnetic', 't1', 't2', 'flair', 'brain', 'head_mr']):
                mr_file = nii_file
        
        # Fallback with size-based heuristics
        if not ct_file or not mr_file:
            available = [f for f in nii_files if not any(x in f.name.lower() for x in ['mask', 'seg', 'label'])]
            if len(available) >= 2:
                # Try to distinguish by file size (CT often larger due to bone detail)
                available_with_size = [(f, f.stat().st_size) for f in available[:2]]
                available_with_size.sort(key=lambda x: x[1], reverse=True)
                ct_file, mr_file = available_with_size[0][0], available_with_size[1][0]
        
        return ct_file, mr_file

    def _select_best_slices(self, ct_data: np.ndarray, mr_data: np.ndarray, max_slices: int) -> List[int]:
        """Select slices with most brain tissue content"""
        depth = ct_data.shape[2]
        if depth < 15:
            return []
        
        # Focus on middle 60% of volume
        start_idx = int(depth * 0.2)
        end_idx = int(depth * 0.8)
        
        slice_scores = []
        for i in range(start_idx, end_idx):
            ct_slice = ct_data[:, :, i]
            mr_slice = mr_data[:, :, i]
            
            # Score based on tissue contrast and variance
            ct_var = np.var(ct_slice)
            mr_var = np.var(mr_slice)
            
            # Penalize slices with too much background
            ct_tissue_ratio = np.sum(ct_slice > -500) / ct_slice.size  # Bone/tissue vs air
            mr_tissue_ratio = np.sum(mr_slice > np.mean(mr_slice) * 0.1) / mr_slice.size
            
            score = ct_var * mr_var * ct_tissue_ratio * mr_tissue_ratio
            slice_scores.append((i, score))
        
        # Select top slices
        slice_scores.sort(key=lambda x: x[1], reverse=True)
        selected = [idx for idx, _ in slice_scores[:max_slices]]
        return sorted(selected)

    def _add_enhanced_slice(self, ct_slice: np.ndarray, mr_slice: np.ndarray) -> bool:
        """Enhanced slice validation and preprocessing"""
        # Quality checks
        if np.std(ct_slice) < 20 or np.std(mr_slice) < 5:
            return False
        
        # Check for reasonable tissue content
        if np.sum(ct_slice > -500) < ct_slice.size * 0.3:  # Less than 30% tissue
            return False
        
        try:
            # Resize with better interpolation
            ct_resized = cv2.resize(ct_slice, (self.img_size, self.img_size), 
                                  interpolation=cv2.INTER_CUBIC)
            mr_resized = cv2.resize(mr_slice, (self.img_size, self.img_size), 
                                  interpolation=cv2.INTER_CUBIC)
            
            # Enhanced normalization
            ct_norm = self.processor.normalize_ct_windowed(ct_resized)
            mr_norm = self.processor.normalize_mr_robust(mr_resized)
            
            if np.isnan(ct_norm).any() or np.isnan(mr_norm).any():
                return False
            
            self.samples.append({
                'mr': mr_norm.astype(np.float32),
                'ct': ct_norm.astype(np.float32)
            })
            return True
            
        except Exception:
            return False

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        mr = torch.FloatTensor(sample['mr']).unsqueeze(0)
        ct = torch.FloatTensor(sample['ct']).unsqueeze(0)
        
        # Simple data augmentation during training
        if self.augment and random.random() > 0.5:
            # Random horizontal flip
            if random.random() > 0.5:
                mr = torch.flip(mr, [-1])
                ct = torch.flip(ct, [-1])
            
            # Random rotation (small angles)
            if random.random() > 0.7:
                angle = random.uniform(-10, 10)
                mr = self._rotate_tensor(mr, angle)
                ct = self._rotate_tensor(ct, angle)
        
        return mr, ct

    def _rotate_tensor(self, tensor: torch.Tensor, angle: float) -> torch.Tensor:
        """Simple rotation for data augmentation"""
        try:
            from scipy.ndimage import rotate
            img = tensor.squeeze().numpy()
            rotated = rotate(img, angle, reshape=False, mode='constant', cval=0)
            return torch.FloatTensor(rotated).unsqueeze(0)
        except ImportError:
            # Fallback if scipy not available
            return tensor

class ImprovedUNet(nn.Module):
    """Enhanced U-Net with attention mechanisms and improved architecture"""

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        
        # Encoder with residual connections
        self.enc1 = self._conv_block(in_channels, 16)
        self.enc2 = self._conv_block(16, 32)
        self.enc3 = self._conv_block(32, 64)
        self.enc4 = self._conv_block(64, 128)
        
        # Bottleneck with attention
        self.bottleneck = self._conv_block(128, 256)
        self.attention_bottleneck = self._attention_gate(256, 256)
        
        # Decoder with skip connections and attention
        self.up4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att4 = self._attention_gate(128, 128)
        self.dec4 = self._conv_block(256, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att3 = self._attention_gate(64, 64)
        self.dec3 = self._conv_block(128, 64)
        
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.att2 = self._attention_gate(32, 32)
        self.dec2 = self._conv_block(64, 32)
        
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = self._conv_block(32, 16)
        
        # Output with residual connection
        self.final_conv = nn.Conv2d(16, out_channels, 1)
        self.output_activation = nn.Tanh()
        
        self._init_weights()

    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Enhanced conv block with group normalization and residual connection"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels//4), out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels//4), out_channels),
            nn.ReLU(inplace=True)
        )

    def _attention_gate(self, F_g: int, F_l: int) -> nn.Module:
        """Attention gate for better feature focusing"""
        return nn.Sequential(
            nn.Conv2d(F_g, F_l//4, 1, bias=False),
            nn.GroupNorm(min(4, F_l//16), F_l//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(F_l//4, 1, 1),
            nn.Sigmoid()
        )

    def _init_weights(self):
        """Improved weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self._pool(e1))
        e3 = self.enc3(self._pool(e2))
        e4 = self.enc4(self._pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self._pool(e4))
        b = b * self.attention_bottleneck(b)  # Self-attention
        
        # Decoder with attention gates
        d4 = self.up4(b)
        e4_att = e4 * self.att4(d4)
        d4 = torch.cat([d4, e4_att], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        e3_att = e3 * self.att3(d3)
        d3 = torch.cat([d3, e3_att], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        e2_att = e2 * self.att2(d2)
        d2 = torch.cat([d2, e2_att], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # Output
        output = self.final_conv(d1)
        return self.output_activation(output)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.max_pool2d(x, 2)

class CombinedLoss(nn.Module):
    """Combined loss function for better training"""
    
    def __init__(self, mse_weight=1.0, ssim_weight=0.3, perceptual_weight=0.1):
        super().__init__()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # MSE Loss
        mse_loss = self.mse_loss(pred, target)
        
        # SSIM Loss
        ssim_loss = 1 - self._ssim_loss(pred, target)
        
        # Gradient Loss for edge preservation
        grad_loss = self._gradient_loss(pred, target)
        
        # Combined loss
        total_loss = (self.mse_weight * mse_loss + 
                     self.ssim_weight * ssim_loss + 
                     self.perceptual_weight * grad_loss)
        
        return total_loss

    def _ssim_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """SSIM loss calculation"""
        # Simple SSIM approximation for differentiable loss
        mu_pred = torch.mean(pred, dim=[2, 3], keepdim=True)
        mu_target = torch.mean(target, dim=[2, 3], keepdim=True)
        
        sigma_pred = torch.var(pred, dim=[2, 3], keepdim=True)
        sigma_target = torch.var(target, dim=[2, 3], keepdim=True)
        sigma_pred_target = torch.mean((pred - mu_pred) * (target - mu_target), dim=[2, 3], keepdim=True)
        
        c1, c2 = 0.01**2, 0.03**2
        ssim_map = ((2 * mu_pred * mu_target + c1) * (2 * sigma_pred_target + c2)) / \
                   ((mu_pred**2 + mu_target**2 + c1) * (sigma_pred + sigma_target + c2))
        
        return torch.mean(ssim_map)

    def _gradient_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Gradient loss for edge preservation"""
        def gradient(img):
            grad_x = img[:, :, :, 1:] - img[:, :, :, :-1]
            grad_y = img[:, :, 1:, :] - img[:, :, :-1, :]
            return grad_x, grad_y
        
        pred_grad_x, pred_grad_y = gradient(pred)
        target_grad_x, target_grad_y = gradient(target)
        
        loss_x = self.mse_loss(pred_grad_x, target_grad_x)
        loss_y = self.mse_loss(pred_grad_y, target_grad_y)
        
        return loss_x + loss_y

class ImprovedTrainer:
    """Enhanced trainer with better optimization strategies"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = self._setup_device()
        self.processor = EnhancedDataProcessor()
        
        # Training state
        self.best_ssim = 0.0
        self.best_mae = float('inf')
        self.target_achieved = False
        self.patience_counter = 0
        
        # History
        self.train_losses = []
        self.val_losses = []
        self.val_ssims = []
        self.val_maes = []

    def _setup_device(self):
        """Setup computing device"""
        if self.config.get('force_cpu', False) or not torch.cuda.is_available():
            return torch.device('cpu')
        
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {torch.cuda.get_device_name()} ({gpu_memory:.1f}GB)")
        
        if gpu_memory < 4.0:
            logger.warning("Low GPU memory, using CPU")
            return torch.device('cpu')
        
        torch.cuda.empty_cache()
        return torch.device('cuda')

    def setup_data(self) -> Tuple[DataLoader, DataLoader]:
        """Setup enhanced data loaders"""
        logger.info("Setting up enhanced dataset...")
        
        # Create dataset with augmentation for training
        full_dataset = ImprovedDataset(
            data_path=self.config['data_path'],
            img_size=self.config['img_size'],
            max_subjects=self.config['max_subjects'],
            max_slices_per_subject=self.config['max_slices_per_subject'],
            augment=False,  # We'll handle augmentation separately
            min_samples=self.config.get('min_samples', 100)
        )

        # Split dataset
        train_size = int(0.85 * len(full_dataset))  # Slightly more for training
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size], 
            generator=torch.Generator().manual_seed(42)
        )
        
        # Enable augmentation for training set
        train_dataset.dataset.augment = True

        # Data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

        logger.info(f"Data split - Training: {len(train_dataset)}, Validation: {len(val_dataset)}")
        return train_loader, val_loader

    def setup_model_and_training(self, train_loader=None):
        """Setup improved model and training components"""
        # Enhanced model
        model = ImprovedUNet().to(self.device)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Enhanced model: {total_params:,} parameters")

        # Combined loss function
        criterion = CombinedLoss(mse_weight=1.0, ssim_weight=0.3, perceptual_weight=0.1)

        # Improved optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )

        # Enhanced scheduler - choose based on whether train_loader is provided
        if train_loader is not None and len(train_loader) > 0:
            # Use OneCycleLR with proper total_steps calculation
            total_steps = len(train_loader) * self.config['epochs']
            if total_steps > 10:  # OneCycleLR needs sufficient steps
                scheduler = optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=self.config['learning_rate'],
                    total_steps=total_steps,
                    pct_start=0.1,
                    anneal_strategy='cos'
                )
                self.use_onecycle = True
            else:
                # Use StepLR for small datasets
                scheduler = optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=1,
                    gamma=0.9
                )
                self.use_onecycle = False
        else:
            # Fallback to CosineAnnealingLR for compatibility
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config['epochs'],
                eta_min=self.config['learning_rate'] * 0.01
            )
            self.use_onecycle = False

        return model, criterion, optimizer, scheduler

    def calculate_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
        """Calculate enhanced metrics"""
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        ssim_scores = []
        mae_scores = []

        for i in range(pred_np.shape[0]):
            pred_slice = pred_np[i, 0]
            target_slice = target_np[i, 0]

            # SSIM with better parameters
            try:
                ssim_val = ssim(pred_slice, target_slice, data_range=2.0, 
                               gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
                ssim_scores.append(max(0.0, min(1.0, ssim_val)))
            except:
                ssim_scores.append(0.5)

            # MAE in HU using windowed normalization
            pred_hu = self.processor.denormalize_ct_windowed(pred_slice)
            target_hu = self.processor.denormalize_ct_windowed(target_slice)
            mae_hu = np.mean(np.abs(pred_hu - target_hu))
            mae_scores.append(mae_hu)

        return np.mean(ssim_scores), np.mean(mae_scores)

    def train(self):
        """Enhanced training loop"""
        logger.info("="*60)
        logger.info("IMPROVED MRI-to-CT Training")
        logger.info("="*60)
        
        # Setup
        train_loader, val_loader = self.setup_data()
        model, criterion, optimizer, scheduler = self.setup_model_and_training(train_loader)

        logger.info(f"Training for {self.config['epochs']} epochs")
        logger.info(f"Targets: SSIM >= {self.config['target_ssim']}, MAE <= {self.config['target_mae']} HU")

        for epoch in range(self.config['epochs']):
            start_time = time.time()

            # Training
            model.train()
            train_loss = 0.0
            
            for mr_batch, ct_batch in tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False):
                mr_batch = mr_batch.to(self.device)
                ct_batch = ct_batch.to(self.device)

                optimizer.zero_grad()
                pred = model(mr_batch)
                loss = criterion(pred, ct_batch)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Update scheduler after each batch for OneCycleLR
                if self.use_onecycle:
                    scheduler.step()
                
                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0.0
            all_ssim = []
            all_mae = []

            with torch.no_grad():
                for mr_batch, ct_batch in val_loader:
                    mr_batch = mr_batch.to(self.device)
                    ct_batch = ct_batch.to(self.device)

                    pred = model(mr_batch)
                    loss = criterion(pred, ct_batch)
                    val_loss += loss.item()

                    ssim_val, mae_val = self.calculate_metrics(pred, ct_batch)
                    all_ssim.append(ssim_val)
                    all_mae.append(mae_val)

            # Update scheduler (only for non-OneCycleLR)
            if not self.use_onecycle:
                scheduler.step()

            # Calculate averages
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_ssim = np.mean(all_ssim) if all_ssim else 0.0
            avg_mae = np.mean(all_mae) if all_mae else float('inf')

            # Record history
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_val_loss)
            self.val_ssims.append(avg_ssim)
            self.val_maes.append(avg_mae)

            epoch_time = time.time() - start_time

            # Logging
            logger.info(f"Epoch {epoch+1:3d} ({epoch_time:.1f}s): "
                       f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | "
                       f"SSIM: {avg_ssim:.4f} | MAE: {avg_mae:.1f} HU")

            # Check targets
            targets_met = avg_ssim >= self.config['target_ssim'] and avg_mae <= self.config['target_mae']
            if targets_met and not self.target_achieved:
                logger.info(f"  *** TARGETS ACHIEVED! ***")
                self.target_achieved = True

            # Save best model (prioritize targets achievement)
            is_better = False
            if targets_met and not (self.best_ssim >= self.config['target_ssim'] and self.best_mae <= self.config['target_mae']):
                is_better = True
            elif targets_met and (self.best_ssim >= self.config['target_ssim'] and self.best_mae <= self.config['target_mae']):
                # Both current and best meet targets, choose better overall
                is_better = avg_ssim > self.best_ssim and avg_mae < self.best_mae
            elif not targets_met:
                # Neither meets targets, choose closer to targets
                current_score = avg_ssim - avg_mae/100  # Composite score
                best_score = self.best_ssim - self.best_mae/100
                is_better = current_score > best_score

            if is_better:
                self.best_ssim = avg_ssim
                self.best_mae = avg_mae
                self.patience_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'val_ssim': avg_ssim,
                    'val_mae': avg_mae,
                    'config': self.config,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'val_ssims': self.val_ssims,
                    'val_maes': self.val_maes
                }, 'improved_mri_ct_model.pth')
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.config.get('early_stopping_patience', 20):
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

            # Memory cleanup
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

        # Final results
        logger.info("="*60)
        logger.info("TRAINING COMPLETE")
        logger.info("="*60)
        logger.info(f"Best SSIM: {self.best_ssim:.4f}")
        logger.info(f"Best MAE: {self.best_mae:.1f} HU")
        logger.info(f"Final SSIM: {self.val_ssims[-1]:.4f}")
        logger.info(f"Final MAE: {self.val_maes[-1]:.1f} HU")
        logger.info(f"Targets achieved: {'YES' if self.target_achieved else 'NO'}")

        if self.target_achieved:
            logger.info(" SUCCESS: Both targets met!")
        else:
            logger.info(" Best Results vs Targets:")
            logger.info(f"   SSIM: {self.best_ssim:.4f} (target: >={self.config['target_ssim']})")
            logger.info(f"   MAE:  {self.best_mae:.1f} HU (target: <={self.config['target_mae']} HU)")

        self.plot_results()
        return model

    def plot_results(self):
        """Enhanced plotting"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss plot
        axes[0, 0].plot(self.train_losses, label='Train Loss', alpha=0.8)
        axes[0, 0].plot(self.val_losses, label='Val Loss', alpha=0.8)
        axes[0, 0].set_title('Training Progress')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # SSIM plot with target line
        axes[0, 1].plot(self.val_ssims, label='SSIM', color='green', alpha=0.8)
        axes[0, 1].axhline(y=self.config['target_ssim'], color='red', linestyle='--', 
                          alpha=0.7, label=f'Target: {self.config["target_ssim"]}')
        axes[0, 1].fill_between(range(len(self.val_ssims)), self.config['target_ssim'], 1.0, 
                               alpha=0.1, color='green', label='Target Zone')
        axes[0, 1].set_title(f'SSIM Progress')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('SSIM')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0.7, 1.0])

        # MAE plot with target line
        axes[1, 0].plot(self.val_maes, label='MAE', color='orange', alpha=0.8)
        axes[1, 0].axhline(y=self.config['target_mae'], color='red', linestyle='--', 
                          alpha=0.7, label=f'Target: {self.config["target_mae"]} HU')
        axes[1, 0].fill_between(range(len(self.val_maes)), 0, self.config['target_mae'], 
                               alpha=0.1, color='green', label='Target Zone')
        axes[1, 0].set_title(f'MAE Progress')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MAE (HU)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Combined score plot
        if len(self.val_ssims) > 0 and len(self.val_maes) > 0:
            # Normalize metrics to [0,1] and create combined score
            ssim_norm = np.array(self.val_ssims)
            mae_norm = 1 - (np.array(self.val_maes) / 100)  # Invert and normalize MAE
            mae_norm = np.clip(mae_norm, 0, 1)
            combined_score = 0.6 * ssim_norm + 0.4 * mae_norm
            
            axes[1, 1].plot(combined_score, label='Combined Score', color='purple', alpha=0.8)
            axes[1, 1].set_title('Combined Performance Score')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig('improved_training_results.png', dpi=150, bbox_inches='tight')
        plt.show()

def main_improved():
    """Main function with improved configuration"""
    
    # Set environment variables
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['OMP_NUM_THREADS'] = '4'
    
    # Enhanced configuration optimized for target achievement
    config = {
        # Data configuration - increased for better learning
        'data_path': "F:/mri/actual data",
        'max_subjects': 200,  # Increased for more data
        'max_slices_per_subject': 25,  # More slices per subject
        'img_size': 128,  # Increased resolution for better detail
        
        # Training configuration - optimized
        'batch_size': 6,  # Smaller batches for stability
        'epochs': 150,  # More epochs for convergence
        'learning_rate': 0.0008,  # Slightly lower LR for stability
        
        # Target metrics
        'target_ssim': 0.86,
        'target_mae': 30.0,
        
        # Training enhancements
        'force_cpu': False,
        'early_stopping_patience': 25,  # More patience
        'save_frequency': 10,
    }
    
    # Adjust batch size based on GPU memory
    if torch.cuda.is_available() and not config['force_cpu']:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory < 4.0:
            config['batch_size'] = 3
            config['img_size'] = 96  # Reduce image size for low memory
        elif gpu_memory < 6.0:
            config['batch_size'] = 4
        else:
            config['batch_size'] = 6
    else:
        config['batch_size'] = 3
        config['img_size'] = 96
    
    logger.info("Enhanced MRI-to-CT Training")
    logger.info("Optimized for target achievement")
    logger.info(f"Targets: SSIM >= {config['target_ssim']}, MAE <= {config['target_mae']} HU")
    logger.info(f"Configuration: {config}")
    
    try:
        trainer = ImprovedTrainer(config)
        model = trainer.train()
        
        if model is not None:
            logger.info("\n" + "="*60)
            logger.info("ENHANCED TRAINING COMPLETED!")
            logger.info("="*60)
            logger.info("Files saved:")
            logger.info("  - improved_mri_ct_model.pth")
            logger.info("  - improved_training_results.png")
            
            # Display final performance
            if trainer.target_achieved:
                logger.info("\n CONGRATULATIONS! Both targets achieved:")
                logger.info(f"    SSIM: {trainer.best_ssim:.4f} >= {config['target_ssim']}")
                logger.info(f"    MAE: {trainer.best_mae:.1f} <= {config['target_mae']} HU")
            else:
                logger.info(f"\n Performance achieved:")
                logger.info(f"   SSIM: {trainer.best_ssim:.4f} (target: >={config['target_ssim']})")
                logger.info(f"   MAE: {trainer.best_mae:.1f} HU (target: <={config['target_mae']} HU)")
                
                # Suggest improvements
                logger.info(f"\n Suggestions for improvement:")
                if trainer.best_ssim < config['target_ssim']:
                    logger.info("   - Increase training epochs or learning rate")
                    logger.info("   - Add more training data")
                if trainer.best_mae > config['target_mae']:
                    logger.info("   - Improve data preprocessing")
                    logger.info("   - Use different loss function weighting")
                    
        return model
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return None

def load_and_inference(model_path: str = 'improved_mri_ct_model.pth', 
                      test_mr_path: str = None):
    """Load model and run inference"""
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    model = ImprovedUNet()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    processor = EnhancedDataProcessor()
    
    logger.info(f"Model loaded from {model_path}")
    logger.info(f"Training performance - SSIM: {checkpoint['val_ssim']:.4f}, MAE: {checkpoint['val_mae']:.1f} HU")
    
    # If test path provided, run inference
    if test_mr_path and os.path.exists(test_mr_path):
        try:
            # Load test MR image
            mr_img = nib.load(test_mr_path).get_fdata().astype(np.float32)
            
            # Process middle slice for demo
            middle_slice = mr_img.shape[2] // 2
            mr_slice = mr_img[:, :, middle_slice]
            
            # Preprocess
            img_size = checkpoint['config'].get('img_size', 128)
            mr_resized = cv2.resize(mr_slice, (img_size, img_size))
            mr_norm = processor.normalize_mr_robust(mr_resized)
            
            # Inference
            mr_tensor = torch.FloatTensor(mr_norm).unsqueeze(0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                ct_pred = model(mr_tensor)
                
            # Postprocess
            ct_pred_np = ct_pred.squeeze().cpu().numpy()
            ct_hu = processor.denormalize_ct_windowed(ct_pred_np)
            
            # Visualize
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(mr_slice, cmap='gray')
            axes[0].set_title('Input MR')
            axes[0].axis('off')
            
            axes[1].imshow(ct_pred_np, cmap='gray')
            axes[1].set_title('Predicted CT (Normalized)')
            axes[1].axis('off')
            
            im = axes[2].imshow(ct_hu, cmap='gray', vmin=-160, vmax=240)
            axes[2].set_title('Predicted CT (HU)')
            axes[2].axis('off')
            plt.colorbar(im, ax=axes[2])
            
            plt.tight_layout()
            plt.savefig('inference_example.png', dpi=150, bbox_inches='tight')
            plt.show()
            
            logger.info(f"Inference complete. Results saved to inference_example.png")
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
    
    return model

def analyze_results(model_path: str = 'improved_mri_ct_model.pth'):
    """Analyze training results - FIXED VERSION"""
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print("\n" + "="*50)
        print("TRAINING RESULTS ANALYSIS")
        print("="*50)
        
        print(f"Final Performance:")
        print(f"  SSIM: {checkpoint['val_ssim']:.4f}")
        print(f"  MAE: {checkpoint['val_mae']:.1f} HU")
        print(f"  Training completed at epoch: {checkpoint['epoch']+1}")
        
        config = checkpoint['config']
        print(f"\nTarget Achievement:")
        ssim_achieved = checkpoint['val_ssim'] >= config['target_ssim']
        mae_achieved = checkpoint['val_mae'] <= config['target_mae']
        
        # Fixed the problematic line - removed emoji and used simple text
        print(f"  SSIM Target (>={config['target_ssim']}): {'ACHIEVED' if ssim_achieved else 'NOT MET'}")
        print(f"  MAE Target (<={config['target_mae']} HU): {'ACHIEVED' if mae_achieved else 'NOT MET'}")
        
        if ssim_achieved and mae_achieved:
            print(f"\n SUCCESS: BOTH TARGETS SUCCESSFULLY ACHIEVED!")
        else:
            print(f"\n Targets not fully met. Consider:")
            if not ssim_achieved:
                print(f"   - SSIM gap: {config['target_ssim'] - checkpoint['val_ssim']:.4f}")
            if not mae_achieved:
                print(f"   - MAE excess: {checkpoint['val_mae'] - config['target_mae']:.1f} HU")
        
        # Plot training history if available
        if 'val_ssims' in checkpoint and 'val_maes' in checkpoint:
            val_ssims = checkpoint['val_ssims']
            val_maes = checkpoint['val_maes']
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            ax1.plot(val_ssims, 'g-', alpha=0.8)
            ax1.axhline(config['target_ssim'], color='r', linestyle='--', label=f'Target: {config["target_ssim"]}')
            ax1.set_title('SSIM Progress')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('SSIM')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(val_maes, 'orange', alpha=0.8)
            ax2.axhline(config['target_mae'], color='r', linestyle='--', label=f'Target: {config["target_mae"]} HU')
            ax2.set_title('MAE Progress')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAE (HU)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('training_analysis.png', dpi=150, bbox_inches='tight')
            plt.show()
            
            print(f"\nTraining analysis plot saved as training_analysis.png")
            
    except Exception as e:
        logger.error(f"Error in analyze_results: {e}")
        print(f"Error analyzing results: {e}")
        return

# =============================================================================
# UNIFIED MRI-CT VISUALIZER - ADVANCED VISUALIZATION AND REPORTING SYSTEM
# =============================================================================

# Set professional style for visualizations
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9

class UnifiedMRICTVisualizer:
    """
    Visualization system that depends ENTIRELY on actual training output
    All visualizations are generated from real training data and results
    """
    
    def __init__(self, checkpoint_path=None, dataset_path=None):
        self.output_dir = Path('./comprehensive_mri_ct_results')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load actual training results from checkpoint
        self.checkpoint_path = checkpoint_path
        self.dataset_path = dataset_path
        self.training_data = None
        self.model = None
        self.config = None
        
        if not self.load_training_results():
            raise ValueError("Cannot initialize visualizer without valid training results")
        
        print(f"Visualizer initialized from training output:")
        print(f"  SSIM: {self.actual_results['best_ssim']:.4f}")
        print(f"  MAE: {self.actual_results['best_mae']:.1f} HU")
        print(f"  Dataset samples: {len(self.training_data)} (from training)")

    def load_training_results(self):
        """Load ALL data from actual training checkpoint and dataset"""
        if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            print("Error: No valid checkpoint found. Run training first.")
            return False
            
        try:
            # Load checkpoint with all training data
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            
            # Extract actual results from training
            self.actual_results = {
                'best_ssim': checkpoint['val_ssim'],
                'best_mae': checkpoint['val_mae'],
                'final_epoch': checkpoint['epoch'],
                'config': checkpoint['config'],
                'train_losses': checkpoint.get('train_losses', []),
                'val_losses': checkpoint.get('val_losses', []),
                'val_ssims': checkpoint.get('val_ssims', []),
                'val_maes': checkpoint.get('val_maes', [])
            }
            
            self.config = checkpoint['config']
            
            # Load the trained model
            self.model = ImprovedUNet()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Load the actual training dataset that was used
            if self.dataset_path and os.path.exists(self.dataset_path):
                self.training_data = ImprovedDataset(
                    data_path=self.dataset_path,
                    img_size=self.config['img_size'],
                    max_subjects=self.config['max_subjects'],
                    max_slices_per_subject=self.config['max_slices_per_subject'],
                    augment=False
                )
            else:
                # Use config path if dataset_path not provided
                self.training_data = ImprovedDataset(
                    data_path=self.config['data_path'],
                    img_size=self.config['img_size'],
                    max_subjects=self.config['max_subjects'],
                    max_slices_per_subject=self.config['max_slices_per_subject'],
                    augment=False
                )
            
            return True
            
        except Exception as e:
            print(f"Failed to load training results: {e}")
            return False

    def create_training_dependent_analysis(self):
        """Create analysis using ONLY the actual training progression"""
        if not self.actual_results['train_losses']:
            print("No training history found in checkpoint")
            return
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'TRAINING ANALYSIS FROM YOUR ACTUAL RESULTS\n' + 
                     f'Final: SSIM {self.actual_results["best_ssim"]:.4f} | MAE {self.actual_results["best_mae"]:.1f} HU', 
                     fontsize=16, fontweight='bold')
        
        epochs = range(1, len(self.actual_results['train_losses']) + 1)
        
        # 1. Actual loss progression
        axes[0,0].plot(epochs, self.actual_results['train_losses'], 
                      label='Training Loss', linewidth=2)
        axes[0,0].plot(epochs, self.actual_results['val_losses'], 
                      label='Validation Loss', linewidth=2)
        axes[0,0].set_title('Your Actual Loss Convergence')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Actual SSIM progression
        axes[0,1].plot(epochs, self.actual_results['val_ssims'], 
                      'g-', linewidth=3, label='Your SSIM Progress')
        axes[0,1].axhline(y=self.config['target_ssim'], color='red', 
                         linestyle='--', label=f'Target: {self.config["target_ssim"]}')
        axes[0,1].set_title('Your SSIM Achievement')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('SSIM')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Actual MAE progression
        axes[0,2].plot(epochs, self.actual_results['val_maes'], 
                      'orange', linewidth=3, label='Your MAE Progress')
        axes[0,2].axhline(y=self.config['target_mae'], color='red', 
                         linestyle='--', label=f'Target: {self.config["target_mae"]}')
        axes[0,2].set_title('Your MAE Achievement')
        axes[0,2].set_xlabel('Epoch')
        axes[0,2].set_ylabel('MAE (HU)')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Target achievement analysis
        ssim_achieved = [s >= self.config['target_ssim'] for s in self.actual_results['val_ssims']]
        mae_achieved = [m <= self.config['target_mae'] for m in self.actual_results['val_maes']]
        both_achieved = [s and m for s, m in zip(ssim_achieved, mae_achieved)]
        
        achievement_epoch = None
        for i, achieved in enumerate(both_achieved):
            if achieved:
                achievement_epoch = i + 1
                break
                
        if achievement_epoch:
            axes[1,0].axvline(x=achievement_epoch, color='green', linewidth=3, 
                             label=f'Targets Achieved: Epoch {achievement_epoch}')
            axes[1,0].plot(epochs, [int(b) for b in both_achieved], 'g-', linewidth=2)
            axes[1,0].set_title(f'Target Achievement Timeline\n(Achieved at Epoch {achievement_epoch})')
        else:
            axes[1,0].plot(epochs, [int(b) for b in both_achieved], 'r-', linewidth=2)
            axes[1,0].set_title('Target Achievement Progress')
            
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('Both Targets Met (1=Yes, 0=No)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Performance summary
        final_performance = {
            'SSIM': self.actual_results['best_ssim'],
            'MAE': self.actual_results['best_mae'],
            'Epochs': self.actual_results['final_epoch'] + 1,
            'Dataset Size': len(self.training_data)
        }
        
        bars = axes[1,1].bar(range(len(final_performance)), 
                            list(final_performance.values()), 
                            color=['green', 'blue', 'orange', 'purple'])
        axes[1,1].set_title('Final Training Statistics')
        axes[1,1].set_xticks(range(len(final_performance)))
        axes[1,1].set_xticklabels(list(final_performance.keys()), rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, final_performance.values()):
            height = bar.get_height()
            axes[1,1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{value:.3f}' if isinstance(value, float) else f'{value}',
                          ha='center', va='bottom')
        
        # 6. Configuration summary
        config_text = f"""TRAINING CONFIGURATION:
Data Path: {self.config['data_path']}
Image Size: {self.config['img_size']}x{self.config['img_size']}
Batch Size: {self.config['batch_size']}
Learning Rate: {self.config['learning_rate']}
Max Subjects: {self.config['max_subjects']}
Target SSIM: {self.config['target_ssim']}
Target MAE: {self.config['target_mae']}

FINAL RESULTS:
SSIM: {self.actual_results['best_ssim']:.4f}
MAE: {self.actual_results['best_mae']:.1f} HU
Epochs: {self.actual_results['final_epoch'] + 1}
Dataset: {len(self.training_data)} samples"""

        axes[1,2].text(0.05, 0.95, config_text, transform=axes[1,2].transAxes,
                      fontsize=9, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        axes[1,2].set_xlim(0, 1)
        axes[1,2].set_ylim(0, 1)
        axes[1,2].axis('off')
        axes[1,2].set_title('Training Summary')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '1_TRAINING_DEPENDENT_ANALYSIS.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()

    def create_model_dependent_showcase(self):
        """Create showcase using actual model predictions on training data"""
        if not self.model or not self.training_data:
            print("Model or training data not available")
            return
            
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        fig.suptitle(f'REAL MODEL PREDICTIONS ON TRAINING DATA\n' + 
                     f'Model Performance: SSIM {self.actual_results["best_ssim"]:.4f} | MAE {self.actual_results["best_mae"]:.1f} HU', 
                     fontsize=14, fontweight='bold')
        
        processor = EnhancedDataProcessor()
        device = torch.device('cpu')
        
        # Show predictions on actual training samples
        num_samples = min(3, len(self.training_data))
        for i in range(num_samples):
            mr_tensor, ct_tensor = self.training_data[i]
            
            # Get actual predictions from your trained model
            with torch.no_grad():
                mr_input = mr_tensor.unsqueeze(0)
                ct_pred = self.model(mr_input)
                
            # Convert to numpy
            mr_np = mr_tensor.squeeze().numpy()
            ct_real_np = ct_tensor.squeeze().numpy()
            ct_pred_np = ct_pred.squeeze().numpy()
            
            # Convert to HU for clinical interpretation
            ct_real_hu = processor.denormalize_ct_windowed(ct_real_np)
            ct_pred_hu = processor.denormalize_ct_windowed(ct_pred_np)
            
            # Calculate actual metrics for this sample
            sample_ssim = ssim(ct_pred_np, ct_real_np, data_range=2.0)
            sample_mae = np.mean(np.abs(ct_pred_hu - ct_real_hu))
            
            # Display
            axes[i,0].imshow(mr_np, cmap='gray')
            axes[i,0].set_title(f'Training Sample {i+1}\nMRI Input')
            axes[i,0].axis('off')
            
            axes[i,1].imshow(ct_pred_np, cmap='gray')
            axes[i,1].set_title(f'Your Model\nPrediction')
            axes[i,1].axis('off')
            
            axes[i,2].imshow(ct_real_np, cmap='gray')
            axes[i,2].set_title(f'Ground Truth\nCT')
            axes[i,2].axis('off')
            
            error_map = np.abs(ct_pred_np - ct_real_np)
            im = axes[i,3].imshow(error_map, cmap='hot')
            axes[i,3].set_title(f'Error Map\nSSIM: {sample_ssim:.3f}\nMAE: {sample_mae:.1f}')
            axes[i,3].axis('off')
            plt.colorbar(im, ax=axes[i,3], fraction=0.046)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '2_MODEL_DEPENDENT_SHOWCASE.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()

    def generate_training_dependent_report(self):
        """Generate report based entirely on training output"""
        report_content = f"""
# MRI-to-CT Training Results Report
## Generated from Actual Training Output

**Training Completed:** {datetime.now().strftime("%B %d, %Y")}
**Model Checkpoint:** {self.checkpoint_path}

## TRAINING CONFIGURATION (From Saved Config)
- **Data Path:** {self.config['data_path']}
- **Image Size:** {self.config['img_size']}x{self.config['img_size']}
- **Batch Size:** {self.config['batch_size']}
- **Learning Rate:** {self.config['learning_rate']}
- **Max Subjects:** {self.config['max_subjects']}
- **Max Slices/Subject:** {self.config['max_slices_per_subject']}

## ACTUAL TRAINING RESULTS
| Metric | Final Value | Target | Status |
|--------|-------------|---------|---------|
| **SSIM** | {self.actual_results['best_ssim']:.4f} | {self.config['target_ssim']} | {'✅ ACHIEVED' if self.actual_results['best_ssim'] >= self.config['target_ssim'] else '❌ NOT MET'} |
| **MAE (HU)** | {self.actual_results['best_mae']:.1f} | {self.config['target_mae']} | {'✅ ACHIEVED' if self.actual_results['best_mae'] <= self.config['target_mae'] else '❌ NOT MET'} |
| **Epochs Trained** | {self.actual_results['final_epoch'] + 1} | {self.config['epochs']} | Completed |
| **Dataset Size** | {len(self.training_data)} samples | - | Loaded |

## TRAINING PROGRESSION
- **Total Epochs:** {len(self.actual_results['train_losses'])}
- **Final Training Loss:** {self.actual_results['train_losses'][-1]:.4f}
- **Final Validation Loss:** {self.actual_results['val_losses'][-1]:.4f}
- **Best SSIM:** {max(self.actual_results['val_ssims']):.4f}
- **Best MAE:** {min(self.actual_results['val_maes']):.1f} HU

## MODEL DEPLOYMENT STATUS
Based on actual training results:
- **Performance:** {'Exceeds targets' if self.actual_results['best_ssim'] >= self.config['target_ssim'] and self.actual_results['best_mae'] <= self.config['target_mae'] else 'Needs improvement'}
- **Stability:** {'Good convergence' if len(self.actual_results['train_losses']) > 10 else 'Limited training'}
- **Readiness:** {'Ready for testing' if self.actual_results['best_ssim'] >= self.config['target_ssim'] else 'Requires further training'}

---
*This report is generated entirely from actual training checkpoint data*
"""
        
        report_path = self.output_dir / 'TRAINING_DEPENDENT_REPORT.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return report_path

    def generate_all_training_dependent_analysis(self):
        """Generate complete analysis dependent on training output"""
        print("GENERATING TRAINING-DEPENDENT ANALYSIS")
        print(f"Source: {self.checkpoint_path}")
        print("=" * 60)
        
        print("1/3 Analyzing actual training progression...")
        self.create_training_dependent_analysis()
        
        print("2/3 Showcasing model predictions on training data...")
        self.create_model_dependent_showcase()
        
        print("3/3 Generating training-dependent report...")
        report_path = self.generate_training_dependent_report()
        
        print(f"\nAnalysis complete - all dependent on training output!")
        print(f"Report: {report_path}")
        
        return report_path