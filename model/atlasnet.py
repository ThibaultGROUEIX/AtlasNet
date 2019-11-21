from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from model_blocks import GetDecoder, Identity
from template import GetTemplate


class Atlasnet(nn.Module):

    def __init__(self, trainer):
        super(Atlasnet, self).__init__(trainer)
        self.opt = trainer.opt
        self.device = trainer.device
        if self.opt.remove_all_batchNorms:
            torch.nn.BatchNorm1d = Identity
            print("Replacing all batchnorms by identities.")


        self.template = [GetTemplate(trainer.opt.start_from, trainer.datasets.dataset_train, device = trainer.device, npoints=trainer.opt.number_points / self.opt.nb_primitives) for i in range(0, trainer.opt.nb_primitives)]
        self.decoder_getter = GetDecoder(bottleneck_size=trainer.opt.bottleneck, input_size=trainer.opt.dim_template, output_size=trainer.opt.dim_template,
                                         decoder_type=trainer.opt.decoder)
        self.decoder = nn.ModuleList([self.decoder_getter() for i in range(0, trainer.opt.nb_primitives)])


    def forward(self, latent_vector):
        input_points = [self.template[i].getRandomPoints() for i in range(self.opt.nb_primitives)]
        input_points = [input_points.expand() for i in range(self.opt.nb_primitives)]

        output_points = torch.cat([self.decoder[i](input_points[i],  latent_vector.unsqueeze(2)).unsqueeze(0) for i in range(0, self.opt.nb_primitives)], dim=0)

        return output_points.contiguous()  # batch, nb_prim, num_point, 3


