import os
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np

from scipy.interpolate import UnivariateSpline
from scipy.spatial.transform import Rotation


def consolidate_metadata(pairs_cache_file,
                         imu_file,
                         data_dir='./data/croco_data',
                         pad_length=50):
    # data directory contains separate subdirectories s0,s1,... for each scene
    # we combine all metadata for each scene into a single file for indexing
    all_image_pairs = []
    all_imu = np.zeros((0,6,50))
    for subdir in os.listdir(data_dir):
        if not os.path.isdir(os.path.join(data_dir, subdir)):
            continue
        scene_num = int(subdir[1:])
        scene_pairs = load_image_pairs_from_metadata(data_dir, scene_num)
        imu = load_trajectories(data_dir, scene_num, scene_pairs, pad_length)
        all_imu = np.vstack([all_imu, imu])
        all_image_pairs.extend(scene_pairs)

    with open(pairs_cache_file, 'w') as f:
        for pair in all_image_pairs:
            f.write(" ".join((str(x) for x in pair)) + "\n")
    with open(imu_file, 'wb') as f:
        np.save(f, all_imu)


def load_pairs_cache_file(pairs_cache_file, data_dir='./data/croco_data'):
    with open(pairs_cache_file, 'r') as f:
        lines = f.read().strip().splitlines()
        pairs = [tuple(x for x in line.split()) for line in lines]
        def to_fname(s, idx):
            return os.path.join(data_dir,
                                f"s{s}",
                                "images",
                                "image%07d.jpg" % int(idx))
        fnames = [(to_fname(p[0], p[1]), to_fname(p[0], p[2])) for p in pairs]
        return fnames


def load_image_pairs_from_metadata(data_dir, scene_num):
    scene_path = os.path.join(data_dir, f"s{scene_num}")
    pairs_file = os.path.join(scene_path, "metadata.txt")
    with open(pairs_file, 'r') as f:
        lines = f.read().strip().splitlines()
        image_pairs = []
        for line in lines:
            p = line.split()
            image_pairs.append((scene_num, int(p[0]), int(p[1])))
    return image_pairs


def load_trajectories(data_dir, scene_num, image_pairs, pad_length=50):
    df_path = os.path.join(data_dir, f"s{scene_num}", "trajectory.csv")
    traj_data = pd.read_csv(df_path)

    t_s = traj_data["tracking_timestamp_us"].values * 1e-5
    p = traj_data[["tx_world_device", "ty_world_device", "tz_world_device"]].values
    q = traj_data[["qx_world_device", "qy_world_device", "qz_world_device", "qw_world_device"]].values

    n = t_s.shape[0]

    # compute acceleration data
    p_splines = [UnivariateSpline(t_s, p[:, i], s=1e-4) for i in range(p.shape[1])]
    accel_x = p_splines[0](t_s, nu=2)
    accel_y = p_splines[1](t_s, nu=2)
    accel_z = p_splines[2](t_s, nu=2)
    accel = np.vstack([accel_x, accel_y, accel_z])

    # compute gyrometer data
    rot = Rotation.from_quat(q, scalar_first = False)
    gyro = []
    gyro.append(np.zeros(3))
    for i in range(len(q) - 1):
        R_i = rot[i]
        R_j = rot[i + 1]

        R_rel = R_i.inv() * R_j
        dt_s = (t_s[i+1] - t_s[i])

        eul_rel = R_rel.as_rotvec("xyz")
        omega = eul_rel / dt_s

        gyro.append(omega)
    gyro = np.array(gyro)
    imu = np.vstack([accel, gyro.T])

    # concatenate all trajectories
    # shape (num trajectories, imu dim (6), length per trajectory)
    traj_imu = [imu[:,p[1]:p[2]] for p in image_pairs]
    traj_imu = [np.pad(t, ((0, 0), (0, pad_length-t.shape[1])), 'constant') for t in traj_imu]
    traj_imu = np.array(traj_imu)
    return traj_imu



class ImageIMUDataset(Dataset):

    def __init__(self,
                 pairs_cache_file,
                 imu_file,
                 data_dir='./data/croco_data/'):
        super().__init__()
        self.data_dir = data_dir
        self.pairs_cache_file = pairs_cache_file
        self.imu_file = imu_file
        self.image_pairs = load_pairs_cache_file(self.pairs_cache_file, data_dir)
        with open(imu_file, 'rb') as f:
            self.imu_data = np.load(f)
        assert self.imu_data.shape[0] == len(self.image_pairs)

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, index):
        im1path, im2path = self.image_pairs[index]
        im1 = Image.open(im1path)
        im2 = Image.open(im2path)
        imu = self.imu_data[index]
        return im1, im2, imu


if __name__ == "__main__":
    pairs_cache_fname = 'pairs_cache.txt'
    imu_fname = 'imu.npy'
    data_dir = './data/croco_data'
    consolidate_metadata(pairs_cache_file=pairs_cache_fname,
                         imu_file=imu_fname,
                         data_dir=data_dir)

    dataset = ImageIMUDataset(pairs_cache_fname, imu_fname, data_dir)
    print(len(dataset))
    print(dataset.imu_data.shape)
    print(dataset[0])

