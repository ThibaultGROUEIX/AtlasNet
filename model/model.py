from model.atlasnet import Atlasnet
from model.model_blocks import PointNet
import torch.nn as nn
import model.resnet as resnet


class EncoderDecoder(nn.Module):
    """
    Wrapper for a encoder and a decoder.
    Author : Thibault Groueix 01.11.2019
    """

    def __init__(self, opt):
        super(EncoderDecoder, self).__init__()
        if opt.SVR:
            self.encoder = resnet.resnet18(pretrained=False, num_classes=opt.bottleneck_size)
        else:
            self.encoder = PointNet(nlatent=opt.bottleneck_size)

        self.decoder = Atlasnet(opt)
        self.to(opt.device)

        if not opt.SVR:
            self.apply(weights_init)  # initialization of the weights
        self.eval()

    def forward(self, x, train=True):
        return self.decoder(self.encoder(x), train=train)

    def generate_mesh(self, x):
        return self.decoder.generate_mesh(self.encoder(x))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
