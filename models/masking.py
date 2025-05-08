# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).


# --------------------------------------------------------
# Masking utils
# --------------------------------------------------------

import torch
import torch.nn as nn    
    
class RandomMask(nn.Module):
    """
    random masking
    """

    def __init__(self, num_patches, mask_ratio):
        super().__init__()
        self.num_patches = num_patches
        self.num_mask = int(mask_ratio * self.num_patches)
    
    def __call__(self, x):
        noise = torch.rand(x.size(0), self.num_patches, device=x.device) 
        argsort = torch.argsort(noise, dim=1) 
        return argsort < self.num_mask


class PaddedSequenceMask(nn.Module):
    """
    random masking
    """

    def __init__(self, padded_length, mask_ratio):
        super().__init__()
        self.padded_length = padded_length
        self.mask_ratio = mask_ratio
    
    def __call__(self, x):
        noise = torch.rand(x.size(0), self.padded_length, device=x.device) 
        return noise < self.mask_ratio
