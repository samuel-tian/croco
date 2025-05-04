import torch
from models.croco import CroCoNet
from models.aligator import AligatorNet
from PIL import Image
import torchvision.transforms
from torchvision.transforms import ToTensor, Normalize, Compose

from aria.image_imu_dataset import ImageIMUDataset

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() and torch.cuda.device_count()>0 else 'cpu')
    
    # load 224x224 images and transform them to tensor 
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_mean_tensor = torch.tensor(imagenet_mean).view(1,3,1,1).to(device, non_blocking=True)
    imagenet_std = [0.229, 0.224, 0.225]
    imagenet_std_tensor = torch.tensor(imagenet_std).view(1,3,1,1).to(device, non_blocking=True)
    trfs = Compose([ToTensor(), Normalize(mean=imagenet_mean, std=imagenet_std)])
    image1 = trfs(Image.open('assets/Chateau1.png').convert('RGB')).to(device, non_blocking=True).unsqueeze(0)
    image2 = trfs(Image.open('assets/Chateau2.png').convert('RGB')).to(device, non_blocking=True).unsqueeze(0)

    image_imu_dataset = ImageIMUDataset(root_dir='aria/')
    img1, img2, imu = image_imu_dataset[0]
    imu = torch.from_numpy(imu).to(device, non_blocking=True).type(torch.float32).unsqueeze(0)
    print(imu.dtype)
    print(image1.dtype)
    
    # load model 
    ckpt = torch.load('pretrained_models/CroCo_V2_ViTBase_SmallDecoder.pth', 'cpu')
    model = AligatorNet( **ckpt.get('croco_kwargs',{})).to(device)
    model.eval()
    msg = model.load_state_dict(ckpt['model'], strict=False)
    
    # forward 
    with torch.inference_mode():
        out, mask, target = model(image1, image2, imu)

if __name__=="__main__":
    main()
