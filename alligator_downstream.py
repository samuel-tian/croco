import os
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import matplotlib.pyplot as plt
import tqdm

# Dataset and model imports
from nyuv2.nyuv2 import NYUv2
from nyuv2.depth_loss import abs_rel_loss, delta1

from models.alligator_downstream import (
    AlligatorDownstreamMonocularEncoder,
    args_from_ckpt,
)

from models.head_downstream import PixelwiseTaskWithDPT

# ----------------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------------
def preprocess(frame, transform):
    """Apply image transforms and convert depth to tensor."""
    img = transform(frame["image"])
    depth = torch.from_numpy(frame["depth"]).unsqueeze(0).float()
    return {"image": img, "depth": depth}


def freeze_encoder(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True

def load_pretrained_model(ckpt_path: str, freeze: bool = True, head_type: str = "DPT"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = args_from_ckpt(ckpt)

    # Select head
    if head_type == "DPT":
        head = PixelwiseTaskWithDPT()
    else:
        raise ValueError(f"Unknown head_type: {head_type}")

    # Build model
    model = AlligatorDownstreamMonocularEncoder(head, **args)
    model.load_state_dict(ckpt['model_state'], strict=False)

    # if freeze:
    #     freeze_encoder(model)
    model.eval()
    return model


def get_parameter_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    layer_decay: float,
) -> list:
    """
    Create per-layer learning-rate and weight-decay groups with layer-wise decay.
    """
    no_decay = {"bias", "LayerNorm.weight"}
    num_layers = model.enc_depth  # total transformer depth
    param_groups = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Determine layer id
        if 'encoder.layers.' in name:
            layer_id = int(name.split('encoder.layers.')[1].split('.')[0]) + 1
        else:
            layer_id = 0

        # Scale learning rate
        scale = layer_decay ** (num_layers - layer_id)
        lr = base_lr * scale

        # Apply weight decay except for norm/bias
        decay = 0.0 if any(nd in name for nd in no_decay) else weight_decay

        param_groups.append({
            'params': [param],
            'lr': lr,
            'weight_decay': decay,
        })

    return param_groups


def cosine_scheduler(
    optimizer: optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
) -> LambdaLR:
    """
    Cosine decay with linear warmup.
    """
    def lr_lambda(epoch: int):
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)

def get_visualize(rgb, raw_pred, raw_gt, idx):
    rgb_np = np.array(rgb.detach().cpu().squeeze(0).permute(1, 2, 0))
    pred_np = np.array(raw_pred.detach().cpu().squeeze(0).permute(1, 2, 0))
    gt_np = np.array(raw_gt.detach().cpu().squeeze(0).permute(1, 2, 0))
    error_np = np.abs(pred_np - gt_np)

    pred_mask = (pred_np <= 0) | np.isinf(pred_np)
    gt_mask = (gt_np <= 0) | np.isinf(gt_np)
    error_mask = (error_np <= 0) | np.isinf(error_np)

    pred = pred_np.copy()
    pred[pred_mask] = np.nan

    gt = gt_np.copy()
    gt[gt_mask] = np.nan

    error = error_np.copy()
    error[error_mask] = np.nan
    # pred = pred_np
    # gt = gt_np
    # error = error_np

    vmin = min(pred.min(), gt.min())
    vmax = max(pred.max(), gt.max())

    # auto scale depth colormap if not given
    fig, axes = plt.subplots(1,4, figsize=(20,4), gridspec_kw={'width_ratios':[1,1,1,1]})
    titles = ['RGB Image', 'Predicted Depth', 'GT Depth', 'Absolute Error']

    ax = axes[0]
    ax.imshow(rgb_np)
    ax.set_title(titles[0])
    ax.axis('off')

    im1 = axes[1].imshow(pred, cmap='magma', vmin=vmin, vmax=vmax)
    axes[1].set_title(titles[1])
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(gt, cmap='magma', vmin=vmin, vmax=vmax)
    axes[2].set_title(titles[2])
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2])

    im3 = axes[3].imshow(error, cmap='inferno')
    axes[3].set_title(titles[3])
    axes[3].axis('off')
    plt.colorbar(im3, ax=axes[3])

    plt.tight_layout()
    plt.savefig(f"image_{idx}.png")


# ----------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------
def main():
    CKPT_PATH = "nyuv2/training/nyuv2_depth_epoch0141.pth"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = NYUv2("nyuv2/nyu_depth_v2_labeled.mat", split="test")
    # try:
    #     len(ds) == 796
    # except:
    #     print(len(ds))

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
    )

    # Model, optimizer, scheduler, loss
    model = load_pretrained_model(CKPT_PATH, freeze=True, head_type="DPT")
    model.to(DEVICE)
    avg_delta_1 = 0.0

    # Testing
    count = 0
    for raw_img, input, depth in tqdm.tqdm(loader):
        raw_img = raw_img.to("cuda")
        input = input.to("cuda")
        depth = depth.to("cuda")
        pred = model(input)

        delta_1 = delta1(pred, depth)
        avg_delta_1 += delta_1
        count += 1

        # if delta_1 > 0.8: 
        #     get_visualize(raw_img, pred, depth, count)

    avg_delta_1 /= len(loader)
    print(avg_delta_1)

if __name__ == "__main__":
    main()
