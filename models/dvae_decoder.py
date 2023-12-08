import torch.nn as nn

from collections import OrderedDict

from models.decoder_block import DecoderBlock

class DVAEDecoder(nn.Module):
    def __init__(self, config):
        """
        Decoder architecture used in dVAE of Dall-E
        """
        super(DVAEDecoder, self).__init__()

        self.group_count = config.group_count
        self.n_blk_per_group = config.n_blk_per_group
        self.n_layers = self.n_blk_per_group * self.group_count
        self.channel = config.channel
        self.embedding_dim = config.codebook.embedding_dim

        self.decoder = nn.Sequential(OrderedDict([
            ('group_1', nn.Sequential(OrderedDict([
                *[(f'block_{i + 1}', DecoderBlock(self.embedding_dim if i == 0 else 2**(self.group_count-1)*self.channel, 2**(self.group_count-1)*self.channel, self.n_layers)) for i in range(self.n_blk_per_group)],
                ('upsample', nn.Upsample(scale_factor=2, mode='nearest')),
            ]))),

            *[(f'group_{j + 2}', nn.Sequential(OrderedDict([
                *[(f'block_{i + 1}', DecoderBlock(2**(self.group_count-1-j)*self.channel if i == 0 else 2**(self.group_count-2-j)*self.channel, 2**(self.group_count-2-j)*self.channel, self.n_layers)) for i in range(self.n_blk_per_group)],
                ('upsample', nn.Upsample(scale_factor=2, mode='nearest')),
            ]))) for j in range(self.group_count - 2)],

            ('group_last', nn.Sequential(OrderedDict([
                *[(f'block_{i + 1}', DecoderBlock(2*self.channel if i == 0 else self.channel, self.channel, self.n_layers)) for i in range(self.n_blk_per_group)],
            ]))),

            ('output', nn.Sequential(OrderedDict([
                ('relu', nn.ReLU()),
                ('conv', nn.Conv2d(self.channel, 3, 1)),
            ]))),
        ]))
    
    def forward(self, x):        
        return self.decoder(x)
