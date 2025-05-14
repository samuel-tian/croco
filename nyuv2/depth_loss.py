# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
# 
# --------------------------------------------------------
# Criterion to train CroCo
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import torch

def get_mask(raw_pred, raw_gt):
    invalid_pred = (raw_pred <= 0) | torch.isinf(raw_pred) | torch.isnan(raw_pred)
    invalid_gt   = (raw_gt   <= 0) | torch.isinf(raw_gt)   | torch.isnan(raw_gt)

    valid_mask = ~(invalid_pred | invalid_gt)
    return valid_mask

def abs_rel_loss(raw_pred: torch.Tensor, raw_gt: torch.Tensor):
    """
    pred, gt: (N,H,W) or (batch,...) predicted & ground-truth depths
    mask: optional boolean tensor where True indicates valid pixels
    """
    valid_mask = get_mask(raw_pred, raw_gt)
    pred_valid = raw_pred[valid_mask]
    gt_valid   = raw_gt[valid_mask]

    diff = torch.abs(pred_valid - gt_valid)
    rel  = diff / (gt_valid +  1e-6)
    return rel.mean()

def delta1(pred: torch.Tensor,
           gt:   torch.Tensor,):
    if pred.dim() == 4 and pred.size(1) == 1:
        pred = pred.squeeze(1)
    if gt.dim() == 4 and gt.size(1) == 1:
        gt = gt.squeeze(1)

    mask = get_mask(pred, gt)

    pred = pred[mask]
    gt = gt[mask]

    ratio = torch.max(pred / gt, gt / pred)
    delta1 = (ratio < 1.25).float().mean()

    return delta1




