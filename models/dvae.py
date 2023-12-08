from models.dvae_encoder import DVAEEncoder
from models.dvae_decoder import DVAEDecoder

from models.basevq_trainer import BaseVQTrainer

class DVAE(BaseVQTrainer):
    def __init__(self, config, img_dir):
        super(DVAE, self).__init__(config, img_dir)