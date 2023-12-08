import torch.nn as nn

from collections import OrderedDict

from models.encoder_block import EncoderBlock

class DVAEEncoder(nn.Module):
    def __init__(self, config):
        """
        Encoder architecture used in dVAE of Dall-E
        """
        super(DVAEEncoder, self).__init__()

        self.group_count = config.group_count
        self.n_blk_per_group = config.n_blk_per_group
        self.n_layers = self.group_count * self.n_blk_per_group
        self.channel = config.channel
        self.kw = config.kw
        self.num_embeddings = config.codebook.num_embeddings

        self.encoder = nn.Sequential(OrderedDict([
            ('conv_in', nn.Conv2d(in_channels=3, out_channels=self.channel, kernel_size=self.kw, padding=(self.kw - 1) // 2)),            
            ('group_1', nn.Sequential(OrderedDict([
                *[(f'block_{i + 1}', EncoderBlock(self.channel, self.channel, self.n_layers)) for i in range(self.n_blk_per_group)],
                ('pool', nn.MaxPool2d(kernel_size=2)),
            ]))),

            *[(f'group_{j + 2}', nn.Sequential(OrderedDict([
                *[(f'block_{i + 1}', EncoderBlock(2**(j)*self.channel if i == 0 else 2**(j+1)*self.channel, 2**(j+1)*self.channel, self.n_layers)) for i in range(self.n_blk_per_group)],
                ('pool', nn.MaxPool2d(kernel_size=2)),
            ]))) for j in range(self.group_count - 2)],

            ('group_last', nn.Sequential(OrderedDict([
                *[(f'block_{i + 1}', EncoderBlock(2**(self.group_count-2)*self.channel if i == 0 else 2**(self.group_count-1)*self.channel, 2**(self.group_count-1)*self.channel, self.n_layers)) for i in range(self.n_blk_per_group)],
            ]))),     

            ('output', nn.Sequential(OrderedDict([
                ('relu', nn.ReLU()),
            ]))), 
        ]))

        self.proj = nn.Conv2d(2**(self.group_count-1)*self.channel, self.num_embeddings, kernel_size=1, stride=1)

    def forward(self, x):
        return self.proj(self.encoder(x))
