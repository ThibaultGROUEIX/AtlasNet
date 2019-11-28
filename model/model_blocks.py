from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import os


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class PointNetfeat(nn.Module):
    def __init__(self, nlatent=1024, dim_input=3):
        """Encoder"""

        super(PointNetfeat, self).__init__()
        self.dim_input = dim_input
        self.conv1 = torch.nn.Conv1d(dim_input, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, nlatent, 1)
        self.lin1 = nn.Linear(nlatent, nlatent)
        self.lin2 = nn.Linear(nlatent, nlatent)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(nlatent)
        self.bn4 = torch.nn.BatchNorm1d(nlatent)
        self.bn5 = torch.nn.BatchNorm1d(nlatent)

        self.nlatent = nlatent

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, self.nlatent)
        x = F.relu(self.bn4(self.lin1(x).unsqueeze(-1)))
        x = F.relu(self.bn5(self.lin2(x.squeeze(2)).unsqueeze(-1)))
        return x.squeeze(2)


class Mapping2Dto3D(nn.Module):
    def __init__(self, bottleneck_size=2500, input_size=3, dim_output=3):
        self.bottleneck_size = bottleneck_size
        self.input_size = input_size
        self.dim_output = dim_output
        super(Mapping2Dto3D, self).__init__()
        print(
            f"New MLP decoder : [{input_size}x{self.bottleneck_size}] [{self.bottleneck_size}x{self.bottleneck_size // 2}] [{self.bottleneck_size // 2}x{self.bottleneck_size // 4}] [{self.bottleneck_size // 4}x{dim_output}] ")
        self.conv1 = torch.nn.Conv1d(input_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size // 2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size // 2, self.bottleneck_size // 4, 1)
        self.last_conv = torch.nn.Conv1d(self.bottleneck_size // 4, dim_output, 1)

        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size // 2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size // 4)

    def forward(self, x, latent):
        x = self.conv1(x) + latent
        x = F.relu(self.bn1(x))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.last_conv(x)


class Mapping2Dto3DLight(nn.Module):
    def __init__(self, bottleneck_size=2500, input_size=3, dim_output=3):
        self.bottleneck_size = bottleneck_size
        self.input_size = input_size
        self.dim_output = dim_output

        super(Mapping2Dto3DLight, self).__init__()
        print(
            f"New Light MLP decoder : [{input_size}x{self.bottleneck_size}] [{self.bottleneck_size}x{self.bottleneck_size // 4}] [{self.bottleneck_size // 4}x{dim_output}] ")
        self.conv1 = torch.nn.Conv1d(input_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size // 4, 1)
        self.last_conv = torch.nn.Conv1d(self.bottleneck_size // 4, dim_output, 1)

        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size // 4)

    def forward(self, x, latent):
        y = self.conv1(x) + latent
        y = F.relu(self.bn1(y))
        y = F.relu(self.bn2(self.conv2(y)))
        y = self.last_conv(y)
        return y

    def get_regul(self):
        return self.regularization


def GetDecoder(bottleneck_size=1024, input_size=2, output_size=3, decoder_type="Atlasnet"):
    bottleneck_size = bottleneck_size
    input_size = input_size
    output_size = output_size
    decoder_type = decoder_type
    if decoder_type == "AtlasNet":
        decoder = Mapping2Dto3D(bottleneck_size, input_size, output_size)
    elif decoder_type == "AtlasNetLight":
        decoder = Mapping2Dto3DLight(bottleneck_size, input_size, output_size)
    return decoder
