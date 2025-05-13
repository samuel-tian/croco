import os, math
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset

# your dataset builder
from nyuv2.nyuv2 import NYUv2
# your model pieces
from models.croco_downstream import CroCoDownstreamMonocularEncoder, checkpoint_args_from_ckpt
from models.head_downstream import PixelwiseTaskWithDPT
from models.alligator import AlligatorNet
# your loss
from nyuv2.depth_loss import abs_rel_loss
from torch.optim.lr_scheduler import LambdaLR

def preprocess(frame):
    img   = transform(frame["image"])
    depth = torch.from_numpy(frame["depth"]).unsqueeze(0).float()
    return {"image": img, "depth": depth}

def freeze_model(model):
    for child in encoder.children():
        for param in child.parameters():
            param.requires_grad = False

def load_model():
    ckpt = torch.load("pretrained_models/checkpoint-latest.pth", map_location="cpu")
    ckpt_args = checkpoint_args_from_ckpt(ckpt)

    # model_base.load_state_dict(ckpt["model_state"], strict=True, weights_only=False)
    model = CroCoDownstreamMonocularEncoder(head, **ckpt_args.croco_args)
    msg = model.load_state_dict(ckpt['model'], strict=True, weights_only=False)

    freeze_model(model)

    print('head: PixelwiseTaskWithDPT()')
    head = PixelwiseTaskWithDPT()
    model._set_prediction_head(head)
    model.eval()
    model = model.to(device)

def get_param_groups(model, base_lr, weight_decay, layer_decay):
    no_decay = ["bias", "LayerNorm.weight"]
    num_layers = encoder.depth  
    param_groups = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # determine layer index:
        if "encoder.layers." in name:
            layer_id = int(name.split("encoder.layers.")[1].split(".")[0]) + 1
        else:
            layer_id = 0  # head / embeddings
        scale = layer_decay ** (num_layers - layer_id)
        group_decay = 0.0 if any(nd in name for nd in no_decay) else weight_decay
        param_groups.append({
            "params": [param],
            "lr": base_lr * scale,
            "weight_decay": group_decay,
        })
    return param_groups

def cosine_with_warmup(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)

def main():
    BATCH_SIZE     = 16
    BASE_LR        = 3e-5
    BETAS          = (0.9, 0.99)
    WEIGHT_DECAY   = 1e-6
    LAYER_DECAY    = 0.75
    TOTAL_EPOCHS   = 1500
    WARMUP_EPOCHS  = 100
    IMG_SIZE       = (224, 224)
    DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.5,1.0), ratio=(3/4,4/3)),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                            std =[0.229,0.224,0.225]),
    ])

    ds = load_dataset(path="nyuv2.py", name="default", split="train")
    ds = ds.map(preprocess, remove_columns=["image","depth","accelData"])
    ds.set_format(type="torch", columns=["image","depth"])

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    param_groups = get_param_groups(model, BASE_LR, WEIGHT_DECAY, LAYER_DECAY)
    optimizer = optim.Adam(param_groups, betas=BETAS)

    scheduler = cosine_with_warmup(optimizer, WARMUP_EPOCHS, TOTAL_EPOCHS)
    criterion = abs_rel_loss()

    for epoch in range(1, TOTAL_EPOCHS+1):
        model.train()
        running_loss = 0.0

        for batch in loader:
            imgs = batch["image"].to(DEVICE, non_blocking=True)
            depths = batch["depth"].to(DEVICE, non_blocking=True)

            preds = model(imgs)        
            loss = criterion(preds, depths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / len(loader)
        print(f"[Epoch {epoch:4d}/{TOTAL_EPOCHS}]  loss={avg_loss:.4f}")

        if epoch % 50 == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "opt_state":   optimizer.state_dict(),
                "sched_state": scheduler.state_dict(),
            }, f"nyuv2_depth_epoch{epoch:04d}.pth")

if __name__ = "__main__":
    main()