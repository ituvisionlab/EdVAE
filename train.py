import argparse
import os
import datetime
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data.cifar10 import CIFAR10Data

from models.dvae import DVAE

from utils import load_config, save_codes, cos_anneal

"""
These ramps/decays follow DALL-E Appendix A.2 Training https://arxiv.org/abs/2102.12092
"""
class DecayTemperature(pl.Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # The relaxation temperature τ is annealed from 1 to 1/16 over the first 150,000 updates.
        t = np.exp(-0.00001 * trainer.global_step)
        pl_module.quantizer.temp = t

class RampKL(pl.Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # The KL weight β is increased from 0 to 6.6 over the first 5000 updates
        # "We divide the overall loss by 256 × 256 × 3, so that the weight of the KL term
        # becomes β/192, where β is the KL weight."
        # TODO: OpenAI uses 6.6/192 but kinda tricky to do the conversion here... about 5e-4 works for this repo so far... :\
        t = cos_anneal(0, pl_module.model_config.quantizer.max_annealing_steps, 0.0, pl_module.model_config.quantizer.kl_weight, trainer.global_step)
        pl_module.quantizer.kl_weight = t

class DecayLR(pl.Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # The step size is annealed from 1e10−4 to 1.25e10−6 over 1,200,000 updates. I use 3e-4
        t = cos_anneal(0, 50000, pl_module.train_config.lr, 1.25e-6, trainer.global_step)
        for g in pl_module.optimizer.param_groups:
            g['lr'] = t

def train():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)

    args = parser.parse_args()

    config = load_config(args.config_path)

    pl.seed_everything(config.train.seed)

    experiment_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    log_path = os.path.join("./" + config.model.name,
                            config.model.quantizer.name, 
                            config.data.name, experiment_name)
    os.makedirs(log_path, exist_ok=True)

    ckpt_path = os.path.join(log_path, "ckpt")
    os.makedirs(ckpt_path, exist_ok=True)

    code_path = os.path.join(log_path, "codes")
    os.makedirs(code_path, exist_ok=True)

    save_codes(src=".", dst=code_path, cfg=args.config_path)

    logger = TensorBoardLogger(save_dir=log_path, name="logs")

    img_path = os.path.join(log_path, "imgs")
    os.makedirs(img_path, exist_ok=True)

    callbacks = []
    
    callbacks.append(ModelCheckpoint(dirpath=ckpt_path, 
                    save_top_k=10, 
                    monitor='loss', 
                    mode="min",
                    filename="{epoch:02d}-{val_loss:.2f}"))
    if config.train.annealing:
        callbacks.extend([DecayTemperature(), DecayLR(), RampKL()])

    data = eval(config.data.name)(config.data)
    model = eval(config.model.name)(config, img_path)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        num_nodes=1,
        enable_progress_bar=True,
        logger=logger,
        callbacks=callbacks,
        max_steps=config.train.max_steps,
        check_val_every_n_epoch=config.train.check_val_every_n_epoch
        )

    trainer.fit(model, data)
    
if __name__ == "__main__":
    train()
    