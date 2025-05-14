import numpy as np
import h5py
from torch.utils import data
import hashlib
import torch
import pandas as pd
import torchvision.transforms.functional as TF
from torchvision.transforms import ColorJitter, ToPILImage, ToTensor, Normalize, RandomCrop, Compose

class NYUv2(data.Dataset):

    def __init__(self, file, split="train"):
        super().__init__()
        self.data_path = file
        self.split=split

        with h5py.File(self.data_path, 'r') as f:
            # available keys: ['accelData','depths','images','instances','labels','names','namesToIds','rawDepthFilenames','rawDepths','rawRgbFilenames','sceneTypes','scenes',]

            df = pd.read_csv(f"nyuv2/nyuv2_{self.split}.csv")
            df['id'] = df['files'].str.extract(r'_(\d+)\.png')[0].astype(int)
            indices = df['id'].values
            indices -= 1

            self.images = np.array(f['images'])[indices]
            self.depths = np.array(f['depths'])[indices]
            self.accelData = np.array(f['accelData']).T[indices]
            
        self.to_pil = ToPILImage()
        self.to_tensor= ToTensor()
        self.jitter = ColorJitter(0.4, 0.4, 0.4, 0.1)
        self.normalize = Normalize(mean=[0.485,0.456,0.406],
                                std =[0.229,0.224,0.225])

    def __len__(self):
        return len(self.images)

    # def __getitem__(self, index):
    #     rgb_np = np.ascontiguousarray(
    #         np.rot90(self.images[index].transpose(1, 2, 0), -1)
    #     )
    #     depth_np = np.ascontiguousarray(
    #         np.rot90(self.depths[index], -1)
    #     )

    #     rgb_pil = self.to_pil(rgb_np)
    #     depth_pil = self.to_pil(depth_np)

    #     if self.split == "train":
    #         i, j, h, w = RandomCrop.get_params(
    #             rgb_pil, output_size=(224, 224))

    #         rgb_crop = TF.crop(rgb_pil, i, j, h, w)
    #         depth_crop = TF.crop(depth_pil, i, j, h, w)
    #         rgb_crop = self.jitter(rgb_crop)
        
    #     elif self.split == "test":
    #         rgb_crop = TF.center_crop(rgb_pil, (224, 224))
    #         depth_crop = TF.center_crop(depth_pil, (224, 224))

    #     rgb_t = self.to_tensor(rgb_crop)
    #     norm_rgb_t = self.normalize(rgb_t)

    #     depth_t = torch.from_numpy(depth_np)    # shape (H,W)
    #     depth_t = depth_t.unsqueeze(0)

    #     return rgb_t, norm_rgb_t, depth_t
    def __getitem__(self,index):
        rgb_np=np.ascontiguousarray(np.rot90(self.images[index].transpose(1,2,0),-1))
        depth_np=np.ascontiguousarray(np.rot90(self.depths[index],-1))

        rgb_pil = self.to_pil(rgb_np)
        if self.split=="train":
            i,j,h,w = RandomCrop.get_params(rgb_pil,output_size=(224,224))
            rgb_crop = TF.crop(rgb_pil,i,j,h,w)
            rgb_crop = self.jitter(rgb_crop)
            depth_crop_np = depth_np[i:i+h,j:j+w]

        else:
            rgb_crop = TF.center_crop(rgb_pil,(224,224))
            H,W = depth_np.shape; top=(H-224)//2; left=(W-224)//2
            depth_crop_np = depth_np[top:top+224,left:left+224]

        rgb_t = self.to_tensor(rgb_crop)
        norm_rgb_t = self.normalize(rgb_t)
        depth_t = torch.from_numpy(depth_crop_np.astype(np.float32)).unsqueeze(0)
        return rgb_t,norm_rgb_t,depth_t


if __name__=="__main__":
    ds = NYUv2()
    data_path = "nyu_depth_v2_labeled.mat"
    with open(data_path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()


