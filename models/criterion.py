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

class MaskedMSE(torch.nn.Module):

    def __init__(self, norm_pix_loss=False, pix_masked=True):
        """
            norm_pix_loss: normalize each patch by their pixel mean and variance
            masked: compute loss over the masked patches only 
        """
        super().__init__()
        self.norm_pix_loss = norm_pix_loss
        self.pix_masked = pix_masked 

    def forward(self, pred, mask, target):
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        if self.masked:
            loss = (loss * mask).sum() / mask.sum()  # mean loss on masked patches
        else:
            loss = loss.mean()  # mean loss
        return loss

class MaskedAligatorMSE(torch.nn.Module):

    def __init__(self, norm_pix_loss=False, pix_masked=True):
        """
            norm_pix_loss: normalize each patch by their pixel mean and variance
            masked: compute loss over the masked patches only 
        """
        super().__init__()
        self.norm_pix_loss = norm_pix_loss
        self.pix_masked = pix_masked 

    def forward(self, pix_pred, pix_mask, pix_target, imu_pred, imu_mask):
        
        if self.norm_pix_loss:
            pix_mean = pix_target.mean(dim=-1, keepdim=True)
            pix_var = pix_target.var(dim=-1, keepdim=True)
            pix_target = (pix_target - pix_mean) / (pix_var + 1.e-6)**.5
        
        if self.norm_imu_loss:
            imu_mean = imu_target.mean(dim=-1, keepdim=True)
            imu_var = imu_target.var(dim=-1, keepdim=True)
            imu_target = (imu_target - imu_mean) / (imu_var + 1.e-6)**.5

        pix_loss = (pix_pred - pix_target) ** 2
        pix_loss = pix_loss.mean(dim=-1)  # [N, L], mean loss per patch
        if self.pix_masked:
            pix_loss = (pix_loss * pix_mask).sum() / pix_mask.sum()  # mean loss on masked patches
        else:
            pix_loss = pix_loss.mean()  # mean loss
    
        imu_loss = (imu_pred * imu_mask - imu_pred)**2
        imu_loss = imu_loss.mean(dim=-1)

        loss = pix_loss + imu_loss
        return loss
