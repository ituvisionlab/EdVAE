import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import pytorch_lightning as pl

from models.dvae_encoder import DVAEEncoder
from models.dvae_decoder import DVAEDecoder

from models.dirichlet_quantizer import DirichletQuantizer
from models.gumbel_quantizer import GumbelQuantizer

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class BaseVQTrainer(pl.LightningModule):

    def __init__(self, config, img_dir=None):
        super().__init__()

        self.model_config = config.model
        self.train_config = config.train
        self.data_config = config.data

        self.img_dir = img_dir
        
        encoder_name = self.model_config.name + "Encoder"
        self.encoder = eval(encoder_name)(self.model_config)

        decoder_name = self.model_config.name +"Decoder"
        self.decoder = eval(decoder_name)(self.model_config)

        self.apply(weights_init)
        
        self.quantizer = eval(self.model_config.quantizer.name)(codebook_config=self.model_config.codebook,
                                                        quantizer_config=self.model_config.quantizer)  
    
    def forward(self, img, iter):
        """
        Encode image, quantize tensor, reconstruct image
        iter is the current iteration of the training;
        necessary for kl weight annealing
        """
        logits = self.encoder(img)
        z_q, additional_loss, indices, perplexity = self.quantizer(logits, iter)
        reconst_img = self.decoder(z_q)

        return reconst_img, additional_loss, indices, perplexity
    
    def training_step(self, batch, batch_idx):
        img, label = batch
        iter = self.global_step

        reconst_img, additional_loss, indices, perplexity = self.forward(img, iter)        
        # mean squared loss for reconstruction
        mse = torch.mean((reconst_img - img)**2)
        reconst_loss = mse
        loss = reconst_loss + additional_loss

        self.log('reconst_loss_mse', mse, prog_bar=True)
        self.log('additional_loss', additional_loss, prog_bar=True)
        self.log('loss', loss, prog_bar=True)
        self.log('perplexity', perplexity, prog_bar=True)

        if iter % 500 == 0:
            torchvision.utils.save_image(img[0,:,:,:], os.path.join(self.img_dir, str(iter)+"_real.png"), normalize=True, padding=0)
            torchvision.utils.save_image(reconst_img[0,:,:,:], os.path.join(self.img_dir, str(iter)+"_reconst.png"), normalize=True, padding=0)

        return loss
    
    def validation_step(self, batch, batch_idx):
        img, label = batch
        iter = self.global_step
        
        reconst_img, additional_loss, indices, perplexity = self.forward(img, iter)
        
        reconst_loss = torch.mean((reconst_img - img)**2)
        loss = reconst_loss + additional_loss

        self.log('test_reconst_loss', reconst_loss, prog_bar=True)
        self.log('test_additional_loss', additional_loss, prog_bar=False)
        self.log('test_loss', loss, prog_bar=False)
        self.log('test_perplexity', perplexity, prog_bar=True)

        torchvision.utils.save_image(img[0,:,:,:], os.path.join(self.img_dir, str(iter)+"_test_real.png"), normalize=True, padding=0)
        torchvision.utils.save_image(reconst_img[0,:,:,:], os.path.join(self.img_dir, str(iter)+"_test_reconst.png"), normalize=True, padding=0)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.train_config.lr)
        self.optimizer = optimizer

        return optimizer