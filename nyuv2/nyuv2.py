import numpy as np
import h5py
import datasets
from datasets import BuilderConfig, Features, Value, SplitGenerator, Array2D, Image, Sequence
import hashlib

_HOMEPAGE = "https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html"
_URL = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
_FILE_HASH = "520609c519fba3ba5ac58c8fefcc3530"

class NYUv2(datasets.GeneratorBasedBuilder):
    """NYU Depth Dataset V2"""

    VERSION = datasets.Version("1.2.1")

    BUILDER_CONFIGS = [
        BuilderConfig(name="default", version=VERSION, description="Default configuration for NYUv2 dataset"),
    ]

    DEFAULT_CONFIG_NAME = "default"

    def _split_generators(self, dl_manager):
        data_path = dl_manager.download(_URL)

        # Verify file hash
        with open(data_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        if file_hash != _FILE_HASH:
            raise ValueError(
                f"Downloaded file hash '{file_hash}' does not match expected hash '{_FILE_HASH}'. "
                "The downloaded dataset file might be corrupted or modified."
            )

        return [
            SplitGenerator(
                name="train",
                gen_kwargs={
                    "filepath": data_path,
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        with h5py.File(filepath, 'r') as f:
            # available keys: ['accelData','depths','images','instances','labels','names','namesToIds','rawDepthFilenames','rawDepths','rawRgbFilenames','sceneTypes','scenes',]
            images = np.array(f['images'])
            depths = np.array(f['depths'])
            accelData = np.array(f['accelData']).T

        for idx in range(images.shape[0]):
            yield idx, {
                "image": np.rot90(images[idx].transpose(1, 2, 0), -1),
                "depth": np.rot90(depths[idx], -1),
                "accelData": accelData[idx],
            }

    
        


