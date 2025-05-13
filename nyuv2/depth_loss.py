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

def abs_rel_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor = None):
    """
    pred, gt: (N,H,W) or (batch,...) predicted & ground-truth depths
    mask: optional boolean tensor where True indicates valid pixels
    """
    if mask is None:
        mask = gt > 0  # assume >0 is valid
    diff = torch.abs(pred[mask] - gt[mask])
    rel = diff / gt[mask]
    return rel.mean()

