from omegaconf import OmegaConf
import os
import shutil
import math

def load_config(config_path):
    """
    Load the config by the given path
    """
    config = OmegaConf.load(config_path)

    return config

def save_codes(src, dst, cfg):
    for root, dirs, files in os.walk(src):
        if root in [".", "./data", "./models"]:
            new_dst = os.path.join(dst, root)
            os.makedirs(new_dst, exist_ok=True)
            for file in files:
                if file.endswith(".py"):
                    path = os.path.join(root, file)
                    shutil.copy(path, new_dst)
    shutil.copy(cfg, dst)

# ref: https://github.com/karpathy/deep-vector-quantization/blob/main/dvq/vqvae.py
def cos_anneal(e0, e1, t0, t1, e):
    """ ramp from (e0, t0) -> (e1, t1) through a cosine schedule based on e \in [e0, e1] """
    alpha = max(0, min(1, (e - e0) / (e1 - e0))) # what fraction of the way through are we
    alpha = 1.0 - math.cos(alpha * math.pi/2) # warp through cosine
    t = alpha * t1 + (1 - alpha) * t0 # interpolate accordingly
    return t