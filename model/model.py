from model.atlasnet import Atlasnet
from model.model_blocks import PointNetfeat
import torch.nn as nn
import model.resnet as resnet


class EncoderDecoder(nn.Module):
    def __init__(self, opt):
        super(EncoderDecoder, self).__init__()
        if opt.SVR:
            self.encoder = resnet.resnet18(pretrained=False, num_classes=1024)
        else:
            self.encoder = PointNetfeat(nlatent=opt.bottleneck_size)

        self.decoder = Atlasnet(opt)

    def forward(self, x, train=True):
        return self.decoder(self.encoder(x), train=train)
