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

import socket
import torch.distributed as dist
from torch.multiprocessing import Process

# Dataset and model imports
from datasets import load_dataset
from nyuv2.nyuv2 import NYUv2
from nyuv2.depth_loss import abs_rel_loss

from models.alligator_downstream import (
    AlligatorDownstreamMonocularEncoder,
    args_from_ckpt,
)

from models.head_downstream import PixelwiseTaskWithDPT

def freeze_encoder(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True

def load_pretrained_model(ckpt_path: str, freeze: bool = True, head_type: str = "DPT"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    s = ckpt['args'].model
    args = eval('dict'+s[len('AlligatorNet'):])

    # Select head
    if head_type == "DPT":
        head = PixelwiseTaskWithDPT()
    else:
        raise ValueError(f"Unknown head_type: {head_type}")

    # Build model
    model = AlligatorDownstreamMonocularEncoder(head, **args)
    model.load_state_dict(ckpt['model'], strict=False)

    if freeze:
        freeze_encoder(model)
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

def get_visualize(rgb, raw_pred, raw_gt):
    pred_mask = (raw_pred <= 0) | np.isinf(raw_pred)
    gt_mask   = (raw_gt <= 0) | np.isinf(raw_gt)
    
    raw_error = np.abs(raw_pred - raw_gt)
    error_mask = (raw_error <= 0) | np.isinf(raw_error)

    pred = raw_pred[~pred_mask]
    gt = raw_gt[~gt_mask]
    error = error[~error_mask]

    # auto scale depth colormap if not given
    if vmin is None: vmin = min(pred.min(), gt.min())
    if vmax is None: vmax = max(pred.max(), gt.max())

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    titles = ['RGB Image', 'Predicted Depth', 'GT Depth', 'Absolute Error']

    ax = axes[0]
    ax.imshow(rgb)
    ax.set_title(titles[0])
    ax.axis('off')

    im1 = axes[1].imshow(pred, cmap='magma', vmin=vmin, vmax=vmax)
    axes[1].set_title(titles[1])
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(gt, cmap='magma', vmin=vmin, vmax=vmax)
    axes[2].set_title(titles[2])
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    im3 = axes[3].imshow(error, cmap='inferno')
    axes[3].set_title(titles[3])
    axes[3].axis('off')
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

def get_dist_env():
    if 'OMPI_COMM_WORLD_SIZE' in os.environ:
        world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE'))
    else:
        world_size = int(os.getenv('SLURM_NTASKS'))

    if 'OMPI_COMM_WORLD_RANK' in os.environ:
        global_rank = int(os.getenv('OMPI_COMM_WORLD_RANK'))
    else:
        global_rank = int(os.getenv('SLURM_PROCID'))
    return global_rank, world_size


def main():
    BATCH_SIZE = 16
    BASE_LR = 3e-5
    BETAS = (0.9, 0.99)
    WEIGHT_DECAY = 1e-6
    LAYER_DECAY = 0.75
    TOTAL_EPOCHS = 1500
    WARMUP_EPOCHS = 100
    CKPT_PATH = "pretrained_models/checkpoint-last.pth"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transform

    # ds = load_dataset("nyuv2/nyuv2.py", name="default", split="train", trust_remote_code=True)

    # ds = ds.map(lambda f: preprocess(f, transform), remove_columns=["image", "depth", "accelData"])
    # ds.set_format(type="torch", columns=["image", "depth"])
    
    ds = NYUv2("nyuv2/nyu_depth_v2_labeled.mat", split="train")
    try:
        len(ds) == 796
    except:
        print(len(ds))

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )

    # Model, optimizer, scheduler, loss
    model = load_pretrained_model(CKPT_PATH, freeze=True, head_type="DPT")
    model.to(DEVICE)

    param_groups = get_parameter_groups(model, BASE_LR, WEIGHT_DECAY, LAYER_DECAY)
    optimizer = optim.Adam(param_groups, betas=BETAS)
    scheduler = cosine_scheduler(optimizer, WARMUP_EPOCHS, TOTAL_EPOCHS)
    criterion = abs_rel_loss

    losses = []
    best_loss = float('inf')

    # Training
    for epoch in range(1, TOTAL_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for _, img, depth in loader:
            img = img.to(DEVICE)
            depth = depth.to(DEVICE)
            pred = model(img)
            loss = criterion(pred, depth)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{TOTAL_EPOCHS} — Loss: {avg_loss:.4f}")
        losses.append(np.array(avg_loss))

        # Save checkpoints periodically
        if avg_loss < best_loss:
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "opt_state": optimizer.state_dict(),
                "sched_state": scheduler.state_dict(),
            }
            torch.save(ckpt, f"nyuv2/training/nyuv2_depth_epoch{epoch:04d}.pth") 
            best_loss = avg_loss

        if epoch % 10 == 0 :
            losses_np = np.array(losses)
            np.savetxt("nyuv2/training/losses.txt", losses_np)

if __name__ == "__main__":
    # # Dont change the following :
    # global_rank, world_size = get_dist_env()
    
    # hostname = socket.gethostname()

    # # You have run dist.init_process_group to initialize the distributed environment
    # # Always use NCCL as the backend. Gloo performance is pretty bad and MPI is currently
    # # unsupported (for a number of reasons).     
    # dist.init_process_group(backend='nccl', rank=global_rank, world_size=world_size)

    # # now run your distributed training code
    # run(global_rank, world_size, hostname)
    # main()
    main()

