import os
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np

from scipy.interpolate import UnivariateSpline
from scipy.spatial.transform import Rotation


def consolidate_metadata(pairs_cache_file,
                         imu_file,
                         imu_lengths_file,
                         data_dir='./data/croco_data',
                         pad_length=50):
    # data directory contains separate subdirectories s0,s1,... for each scene
    # we combine all metadata for each scene into a single file for indexing
    all_image_pairs = []
    all_imu = np.zeros((0,50,7))
    all_imu_lengths = np.zeros((0))
    for subdir in os.listdir(data_dir):
        if not os.path.isdir(os.path.join(data_dir, subdir)):
            continue
        scene_num = int(subdir[1:])
        scene_pairs = load_image_pairs_from_metadata(data_dir, scene_num)
        imu, imu_lengths = load_trajectories(data_dir, scene_num, scene_pairs, pad_length)
        all_imu = np.vstack([all_imu, imu])
        all_imu_lengths = np.concat([all_imu_lengths, imu_lengths])
        all_image_pairs.extend(scene_pairs)

    with open(pairs_cache_file, 'w') as f:
        for pair in all_image_pairs:
            f.write(" ".join((str(x) for x in pair)) + "\n")
    with open(imu_file, 'wb') as f:
        np.save(f, all_imu)
    with open(imu_lengths_file, 'wb') as f:
        np.save(f, all_imu_lengths)


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
    accel = accel.T

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

    imu = np.hstack([accel, gyro, np.expand_dims(t_s, axis=-1)])

    # concatenate all trajectories
    # shape (num trajectories, trajectory length, imu dim (7))
    traj_imu = [imu[p[1]:p[2],:] for p in image_pairs]
    traj_lengths = [p[2]-p[1]+1 for p in image_pairs]
    traj_imu = [np.pad(t, ((0, pad_length-t.shape[0]), (0, 0)), 'constant') for t in traj_imu]
    traj_imu = np.array(traj_imu)
    traj_lengths = np.array(traj_lengths)
    return traj_imu, traj_lengths



class ImageIMUDataset(Dataset):

    def __init__(self,
                 root_dir='.',
                 pairs_cache_file='pairs_cache.txt',
                 imu_file='imu.npy',
                 imu_lengths_file='imu_lengths.npy',
                 data_dir='./data/croco_data/'):
        super().__init__()
        self.data_dir = os.path.join(root_dir, data_dir)
        self.pairs_cache_file = os.path.join(root_dir, pairs_cache_file)
        self.imu_file = os.path.join(root_dir, imu_file)
        self.imu_lengths_file = os.path.join(root_dir, imu_lengths_file)
        self.image_pairs = load_pairs_cache_file(self.pairs_cache_file, self.data_dir)
        with open(self.imu_file, 'rb') as f:
            self.imu_data = np.load(f)
        with open(self.imu_lengths_file, 'rb') as f:
            self.imu_lengths = np.load(f)
        assert self.imu_data.shape[0] == len(self.image_pairs)
        assert self.imu_data.shape[0] == self.imu_lengths.shape[0]
        assert len(self.imu_lengths.shape) == 1

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, index):
        im1path, im2path = self.image_pairs[index]
        im1 = Image.open(im1path)
        im2 = Image.open(im2path)
        imu = self.imu_data[index]
        imu_lengths = self.imu_lengths[index:index+1]
        return im1, im2, imu, imu_lengths


if __name__ == "__main__":
    pairs_cache_fname = 'pairs_cache.txt'
    imu_fname = 'imu.npy'
    imu_lengths_fname = 'imu_lengths.npy'
    data_dir = './data/croco_data'
    consolidate_metadata(pairs_cache_file=pairs_cache_fname,
                         imu_file=imu_fname,
                         imu_lengths_file=imu_lengths_fname,
                         data_dir=data_dir)

    dataset = ImageIMUDataset(
            root_dir='.',
            pairs_cache_file=pairs_cache_fname,
            imu_file=imu_fname,
            imu_lengths_file=imu_lengths_fname,
            data_dir=data_dir)

