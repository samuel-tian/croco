import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import shutil
from tqdm import tqdm

import projectaria_tools.core.mps as mps
from projectaria_tools.core.mps.utils import filter_points_from_confidence
from collections import OrderedDict


def get_timestamps(obs_df):
    ts = obs_df['frame_tracking_timestamp_us'].to_list()
    ts = list(OrderedDict.fromkeys(ts))
    return ts
    
def build_uids_cache(obs_df):
    uids_cache = {}
    grouped = obs_df.groupby('frame_tracking_timestamp_us')
    for ts, group in tqdm(grouped, desc="building uid cache"):
        uids_cache[ts] = set(group['uid'].to_list())
    return uids_cache

def compute_covisibility(obs_df, t1_idx, t2_idx, timestamps, uids_cache):
    t1, t2 = timestamps[t1_idx], timestamps[t2_idx]
    t1_uids = uids_cache[t1]
    t2_uids = uids_cache[t2]

    overlap = len(t1_uids & t2_uids)  # fast set intersection
    if len(t2_uids) == 0:
        return 0, 0
    return overlap / len(t2_uids), overlap
    #t1, t2 = timestamps[t1_idx], timestamps[t2_idx]

    #if t1 not in uids_cache:
    #    df_t1 = obs_df.query(f'frame_tracking_timestamp_us == {t1}')
    #    uids_cache[t1] = df_t1['uid'].to_list()
    #if t2 not in uids_cache:
    #    df_t2 = obs_df.query(f'frame_tracking_timestamp_us == {t2}')
    #    uids_cache[t2] = df_t2['uid'].to_list()

    #t1_uids = uids_cache[t1]
    #t2_uids = uids_cache[t2]
    #t1_uids_set = {uid for uid in t1_uids}

    #overlap = 0
    #for uid in t2_uids:
    #    if uid in t1_uids_set:
    #        overlap += 1
    #return overlap / len(t2_uids), overlap


def visualize_covisibility(obs_df, good_pairs):
    n = 15
    fig, axes = plt.subplots(n, 2, figsize=(6, n*3))
    interval = 50
    start = 200
    for j in range(n):
        for i in range(10000):
            rand_idx = np.random.randint(1, len(good_pairs)-1)
            pair = good_pairs[rand_idx]
            idx1, idx2, vis, overlap = pair
            if np.abs(idx2 - idx1) < 10:
                continue
            if overlap > start+interval*j  and overlap < start+interval*j+interval:
                t1, t2 = ts_cache[idx1], ts_cache[idx2]
                fname1, fname2 = "%07d" % (t1 / 1e5), "%07d" % (t2 / 1e5)
                img1 = plt.imread(f"rgb/vignette{fname1}.jpg")
                img2 = plt.imread(f"rgb/vignette{fname2}.jpg")
                axes[j, 0].imshow(img1)
                axes[j, 1].imshow(img2)
                axes[j, 0].set_title(f"{overlap},%.3f co-visibility" % (vis))
                print(t1, t2, fname1, fname2)
                print(pair)
                break
    fig.set_tight_layout(True)
    plt.show()

def sample_and_save_pairs(good_pairs,
                          obs_df,
                          timestamps,
                          in_path,
                          out_path,
                          min_traj_length = 5,
                          min_overlap = 300,
                          max_overlap = 700,
                          num_samples = 50):
    n = len(timestamps)
    found_pairs = set()
    found_pairs_list = []
    for i in tqdm(range(num_samples), desc="sampling pairs"):
        while True:
            rand_idx = np.random.randint(0, len(good_pairs) - 1)
            pair = good_pairs[rand_idx]
            idx1, idx2, vis, overlap = pair

            if np.abs(idx2 - idx1) < min_traj_length:
                continue
            if overlap > max_overlap or overlap < min_overlap:
                continue
            if (idx1, idx2) in found_pairs:
                continue
            
            t1, t2 = int(timestamps[idx1]/1e5), int(timestamps[idx2]/1e5)
            fname1, fname2 = "%07d" % (t1), "%07d" % (t2)
            img1 = os.path.join("rgb", f"vignette{fname1}.jpg")
            img2 = os.path.join("rgb", f"vignette{fname2}.jpg")
            img1_out = os.path.join("images", f"image{fname1}.jpg")
            img2_out = os.path.join("images", f"image{fname2}.jpg")

            out_pair = (t1, t2, pair[2], pair[3])
            shutil.copyfile(os.path.join(in_path, img1),
                            os.path.join(out_path, img1_out))
            shutil.copyfile(os.path.join(in_path, img2),
                            os.path.join(out_path, img2_out))
            found_pairs.add(out_pair)
            found_pairs_list.append(out_pair)
            break

    shutil.copy(os.path.join(in_path, "trajectory.csv"), out_path)
    with open(os.path.join(out_path, "metadata.txt"), 'w') as f:
        for pair in found_pairs:
            f.write(f"{pair[0]} {pair[1]} {pair[2]} {pair[3]}\n")


def generate_and_save_pairs(obs_df,
                            timestamps,
                            obs_path,
                            pickle_path,
                            min_traj_length = 1,
                            min_overlap = 100,
                            max_overlap = 1000):
    n = len(timestamps)
    if not Path(pickle_path).is_file():
        rmdir(Path(obs_path))
        uids_cache = {}
        good_pairs = []
        uids_cache = build_uids_cache(obs_df)
        for i in tqdm(range(n), desc="generating pairs"):
            for j in range(i+min_traj_length, n):
                vis, overlap = compute_covisibility(obs_df, i, j, timestamps, uids_cache)
                if overlap > min_overlap and overlap < max_overlap:
                    good_pairs.append((i, j, vis, overlap))
        with open(pickle_path, 'ab') as f:
            pickle.dump(good_pairs, f)
        return good_pairs

    with open(pickle_path, 'rb') as f:
        good_pairs = pickle.load(f)
        return good_pairs


def fast_generate(obs_df,
                  timestamps,
                  in_path,
                  pickle_path,
                  out_path,
                  min_traj_length=10,
                  max_traj_length=50,
                  min_overlap=300,
                  max_overlap=700,
                  num_samples=50):
    n = len(timestamps)
    uids_cache = build_uids_cache(obs_df)
    found_pairs = set()
    found_pairs_list = []
    num_fails = 0
    for i in tqdm(range(num_samples), desc="sampling pairs"):
        while True:
            if num_fails > 1000:
                num_fails = 0
                min_traj_length = 5
                min_overlap -= 100 
            num_fails += 1

            idx1 = np.random.randint(0, len(timestamps)-1)
            idx2 = idx1 + np.random.randint(min_traj_length, max_traj_length)
            if idx2 >= len(timestamps):
                continue
            vis, overlap = compute_covisibility(obs_df, idx1, idx2, timestamps, uids_cache)
            if overlap < min_overlap or overlap > max_overlap:
                continue

            # if np.abs(idx2 - idx1) < min_traj_length:
            #     continue
            # if overlap > max_overlap or overlap < min_overlap:
            #     continue
            if (idx1, idx2) in found_pairs:
                continue
            
            t1, t2 = int(timestamps[idx1]/1e5), int(timestamps[idx2]/1e5)
            fname1, fname2 = "%07d" % (t1), "%07d" % (t2)
            img1 = os.path.join("rgb", f"vignette{fname1}.jpg")
            img2 = os.path.join("rgb", f"vignette{fname2}.jpg")
            img1_out = os.path.join("images", f"image{fname1}.jpg")
            img2_out = os.path.join("images", f"image{fname2}.jpg")

            out_pair = (t1, t2, vis, overlap)
            shutil.copyfile(os.path.join(in_path, img1),
                            os.path.join(out_path, img1_out))
            shutil.copyfile(os.path.join(in_path, img2),
                            os.path.join(out_path, img2_out))
            found_pairs.add(out_pair)
            found_pairs_list.append(out_pair)

            num_fails = 0
            break

    shutil.copy(os.path.join(in_path, "trajectory.csv"), out_path)
    with open(os.path.join(out_path, "metadata.txt"), 'w') as f:
        for pair in found_pairs:
            f.write(f"{pair[0]} {pair[1]} {pair[2]} {pair[3]}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--scene")
    parser.add_argument("-r", "--root")
    args = parser.parse_args()
    scene = args.scene
    root_dir = args.root

    scene_path = os.path.join(args.root, "tmp_data", args.scene)
    out_path = os.path.join(args.root, f"s{args.scene}")
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(os.path.join(out_path, "images"), exist_ok=True)

    obs_path = os.path.join(scene_path, "semidense_observations.csv.gz")
    obs_df = pd.read_csv(obs_path, compression="gzip")
    ts = get_timestamps(obs_df)

    pickle_path = os.path.join(out_path, f"good_pairs_{scene}.pickle")
    # good_pairs = generate_and_save_pairs(obs_df, ts, out_path, pickle_path)
    # sample_and_save_pairs(good_pairs, obs_df, ts, scene_path, out_path)
    fast_generate(obs_df, ts, scene_path, pickle_path, out_path)


    #print(compute_covisibility(semidense_obs_df, 0, 44))

