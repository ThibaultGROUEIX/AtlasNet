from atlasnet import Atlasnet
from model_blocks import PointNetfeat
import torch.nn as nn
import resnet


class EncoderDecoder(nn.Module):
    def __init__(self, trainer):
        super(EncoderDecoder, self).__init__()
        if trainer.opt.input_type == 'image':
            self.encoder = resnet.resnet18(pretrained=self.pretrained_encoder, num_classes=1024)
        else:
            self.encoder = PointNetfeat( nlatent=trainer.opt.bottleneck_size)

        self.decoder = Atlasnet(trainer)


    def forward(self, x):
        return self.decoder(self.encoder(x))
