"""
Evaluation metrics utilities for vector prediction tasks.
"""

import numpy as np
import torch
from typing import Dict, Union


def compute_vector_errors(preds: Union[np.ndarray, torch.Tensor], 
                         Y: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
    """
    Compute multiple vector prediction metrics.

    Args:
        preds: [N, D] predicted vectors
        Y: [N, D] ground truth vectors
        
    Returns:
        dict with metrics including:
        - L1: Mean absolute error
        - L1_GT: Mean absolute error of predicting mean
        - L2: Mean squared error  
        - L2_GT: Mean squared error of predicting mean
        - EPE: End-point error (mean Euclidean distance)
        - EPE_GT: End-point error of predicting mean
        - 1px_acc, 2px_acc, 3px_acc: Accuracy within pixel thresholds
        - angular_error_deg: Mean angular error in degrees
        - angular_error_deg_GT: Angular error of predicting mean
    """
    # Convert to numpy if needed
    if torch.is_tensor(preds):
        preds = preds.cpu().numpy()
    if torch.is_tensor(Y):
        Y = Y.cpu().numpy()

    # Endpoint errors
    errors = np.linalg.norm(preds - Y, axis=1)
    
    # L1 metrics
    L1 = np.mean(np.abs(preds - Y))
    L1_GT = np.mean(np.abs(np.mean(Y, axis=0) - Y))
    
    # L2 metrics
    L2 = np.mean((preds - Y)**2)
    L2_GT = np.mean((np.mean(Y, axis=0) - Y)**2)
    
    # Endpoint error metrics
    EPE = np.mean(errors)
    EPE_GT = np.mean(np.linalg.norm(np.mean(Y, axis=0) - Y, axis=1))
    
    # Accuracy metrics (fraction within pixel thresholds)
    acc_1px = np.mean(errors < 1.0)
    acc_2px = np.mean(errors < 2.0)
    acc_3px = np.mean(errors < 3.0)

    # Angular error for predictions
    dot = np.sum(preds * Y, axis=1)
    norm_pred = np.linalg.norm(preds, axis=1)
    norm_true = np.linalg.norm(Y, axis=1)
    eps = 1e-8
    cosine = dot / (norm_pred * norm_true + eps)
    cosine = np.clip(cosine, -1.0, 1.0)
    angular_error = np.degrees(np.arccos(cosine))
    mean_angle = np.mean(angular_error)
    
    # Angular error for predicting mean (baseline)
    Y_mean = np.mean(Y, axis=0, keepdims=True).repeat(Y.shape[0], axis=0)
    dot_gt = np.sum(Y_mean * Y, axis=1)
    norm_Y_mean = np.linalg.norm(Y_mean, axis=1)
    cosine_gt = dot_gt / (norm_Y_mean * norm_true + eps)
    cosine_gt = np.clip(cosine_gt, -1.0, 1.0)
    angular_error_gt = np.degrees(np.arccos(cosine_gt))
    mean_angle_GT = np.mean(angular_error_gt)

    return {
        "L1": L1,
        "L1_GT": L1_GT,
        "L2": L2,
        "L2_GT": L2_GT,
        "EPE": EPE,
        "EPE_GT": EPE_GT,
        "1px_acc": acc_1px,
        "2px_acc": acc_2px,
        "3px_acc": acc_3px,
        "angular_error_deg": mean_angle,
        "angular_error_deg_GT": mean_angle_GT
    }