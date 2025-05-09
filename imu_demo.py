import torch
from models.croco import CroCoNet
from models.aligator import AligatorNet
from PIL import Image
import torchvision.transforms
from torchvision.transforms import ToTensor, Normalize, Compose

from torchvision.transforms import v2

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
    img1, img2, imu, imu_length = image_imu_dataset[0]

    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize(size=(224, 224)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])

    imu = torch.from_numpy(imu).to(device, non_blocking=True).type(torch.float32).unsqueeze(0)
    imu_length = torch.from_numpy(imu_length).to(device, non_blocking=True).type(torch.int32).unsqueeze(0)
    img1 = transforms(img1).to(device, non_blocking=True).unsqueeze(0)
    img2 = transforms(img2).to(device, non_blocking=True).unsqueeze(0)
    
    # load model 
    ckpt = torch.load('pretrained_models/CroCo_V2_ViTBase_SmallDecoder.pth', 'cpu')
    model = AligatorNet( **ckpt.get('croco_kwargs',{})).to(device)
    model.eval()
    msg = model.load_state_dict(ckpt['model'], strict=False)

    uninitialized_keys = []
    new_state_dict = model.state_dict()
    for key in model.state_dict():
        if key not in ckpt['model']:
            uninitialized_keys.append(key)
        elif ckpt['model'][key].shape != new_state_dict[key].shape:
            uninitialized_keys.append(key)
    for key in uninitialized_keys:
        print(key)
    
    # forward 
    with torch.inference_mode():
        out, out_img, out_imu, mask, target = model(img1, img2, imu, imu_length)

    # the output is normalized, thus use the mean/std of the actual image to go back to RGB space 
    patchified = model.patchify(img1)
    mean = patchified.mean(dim=-1, keepdim=True)
    var = patchified.var(dim=-1, keepdim=True)

    input_image = img1 * imagenet_std_tensor + imagenet_mean_tensor
    ref_image = img2 * imagenet_std_tensor + imagenet_mean_tensor
    image_masks = model.unpatchify(model.patchify(torch.ones_like(ref_image)) * mask[:,:,None])
    masked_input_image = ((1 - image_masks) * input_image)

    # undo imagenet normalization, prepare masked image
    decoded_image = model.unpatchify(out * (var + 1.e-6)**.5 + mean)
    decoded_image = decoded_image * imagenet_std_tensor + imagenet_mean_tensor

    decoded_image_new = model.unpatchify(out_img * (var + 1.e-6)**.5 + mean)
    decoded_image_new = decoded_image_new * imagenet_std_tensor + imagenet_mean_tensor

    # make visualization
    visualization = torch.cat((ref_image, masked_input_image, decoded_image, decoded_image_new, input_image), dim=3) # 4*(B, 3, H, W) -> B, 3, H, W*4
    B, C, H, W = visualization.shape
    visualization = visualization.permute(1, 0, 2, 3).reshape(C, B*H, W)
    visualization = torchvision.transforms.functional.to_pil_image(torch.clamp(visualization, 0, 1))
    fname = "demo_output.png"
    visualization.save(fname)
    print('Visualization save in '+fname)

if __name__=="__main__":
    main()
