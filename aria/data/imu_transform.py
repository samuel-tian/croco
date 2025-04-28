import os
from torch.utils.data import Dataset

from PIL import Image
from datasets.transforms import get_pair_transforms

from datasets.pairs_dataset import PairsDataset, load_image

class ImuPairsDataset(Dataset):
    
    def __init__(self, dnames, ):
        super().__init__(dnames, trfs, totensor, normalize, data_dir)
        
        self.imu_data ls
        = []

    def load_imu_data(self, ):
        return
    def get_trajectory(t0, tf, file):
        traj_data = pd.read_csv(file)
    
        start_idx = traj_data[traj_data["tracking_timestamp_us"] == t0].index[0]
        end_idx = traj_data[traj_data["tracking_timestamp_us"] == tf].index[0]
    
        traj_data = traj_data[start_idx:end_idx]
    
        t_s = traj_data["tracking_timestamp_us"].values * 1e-6 
        p = traj_data[["tx_world_device", "ty_world_device", "tz_world_device"]].values
        q = traj_data[["qx_world_device", "qy_world_device", "qz_world_device", "qw_world_device"]].values

        return t_s, p, q

    def get_accelerometer(t0, tf, file):
        t_s, p, _ = get_trajectory(t0, tf, file)

        p_splines = [UnivariateSpline(t_s, p[:, i], s=1e-4) for i in range(p.shape[1])]
        accel_x = p_splines[0](t_s, nu=2)
        accel_y = p_splines[1](t_s, nu=2)
        accel_z = p_splines[2](t_s, nu=2)

        accel = np.vstack([accel_x, accel_y, accel_z])

        return accel[:, :-1].T, t_s*1e6

    def get_gyrometer(t0, tf, file):
        t_s, _, q = get_trajectory(t0, tf, file)
        rot = Rotation.from_quat(q, scalar_first = False)
    
        gyro = []
        for i in range(len(q) - 1):
            R_i = rot[i]
            R_j = rot[i + 1]

            R_rel = R_i.inv() * R_j
            dt_s = (t_s[i+1] - t_s[i])

            eul_rel = R_rel.as_rotvec("xyz")
            omega = eul_rel / dt_s

            gyro.append(omega)
    
        return np.array(gyro), t_s*1e6

    def get_imu_data(t0, tf, file):
        accel, t_us = get_accelerometer(t0, tf, file)
        gyro, _ = get_gyrometer(t0, tf, file)
        
        return np.hstack([accel, gyro])


    # need to figure out how to include time stamps hahahahahahahahahhahahahah

    def __getitem__(self, index):
        im1path, im2path = self.image_pairs[index]
        im1 = load_image(im1path)
        im2 = load_image(im2path)
        
        if self.transforms is not None: im1, im2 = self.trasnforms(im1, im2)
        imu_data = self.load_imu_data()

        return im1, im2, imu_data
