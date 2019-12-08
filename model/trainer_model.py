import torch
from auxiliary.my_utils import yellow_print
from model.model import EncoderDecoder
import torch.optim as optim
import numpy as np
import torch.nn as nn


class TrainerModel(object):
    def __init__(self):
        """
        This class creates the architectures and implements all trainer functions related to architecture.
        Author : Thibault Groueix 01.11.2019
        """
        super(TrainerModel, self).__init__()

    def build_network(self):
        """
        Create network architecture. Refer to auxiliary.model
        :return:
        """
        if torch.cuda.is_available(): # TODO:
            self.opt.device = torch.device(f"cuda:{self.opt.multi_gpu[0]}")
        else:
            # Run on CPU
            self.opt.device = torch.device(f"cpu")

        self.network = EncoderDecoder(self.opt)
        self.reload_network()
        self.network = nn.DataParallel(self.network, device_ids=self.opt.multi_gpu)

    def reload_network(self):
        """
        Reload entire model or only decoder (atlasnet) depending on the options
        :return:
        """
        if self.opt.reload_model_path != "":
            try:
                self.network.load_state_dict(torch.load(self.opt.reload_model_path))
                yellow_print(f"Network weights loaded from  {self.opt.reload_model_path}!")
            except:
                yellow_print(f"Failed to reload {self.opt.reload_model_path}")

        elif self.opt.reload_decoder_path != "":
            try:
                # 1. filter out unnecessary keys
                model_dict = self.network.decoder.state_dict()
                pretrained_dict = torch.load(self.opt.reload_decoder_path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                # 2. overwrite entries in the existing state dict
                model_dict.update(pretrained_dict)
                self.network.decoder.load_state_dict(model_dict)
                yellow_print(f"Network Decoder weights loaded from  {self.opt.reload_decoder_path}!")
            except:
                yellow_print(f"Failed to reload decoder {self.opt.reload_decoder_path}")
        else:
            yellow_print("No network weights to reload!")

    def build_optimizer(self):
        """
        Create optimizer
        """
        if self.opt.train_only_encoder:
            # To train a resnet image encoder with a pre-trained atlasnet decoder.
            yellow_print("only train the Encoder")
            self.optimizer = optim.Adam(self.network.module.encoder.parameters(), lr=self.opt.lrate)
        else:
            self.optimizer = optim.Adam(self.network.module.parameters(), lr=self.opt.lrate)

        if self.opt.reload_optimizer_path != "":
            try:
                self.optimizer.load_state_dict(torch.load(self.opt.reload_optimizer_path))
                yellow_print(f"Reloaded optimizer {self.opt.reload_optimizer_path}")
            except:
                yellow_print(f"Failed to reload optimizer {self.opt.reload_optimizer_path}")

        # Set policy for warm-up if you use multiple GPUs
        self.next_learning_rates = []
        if len(self.opt.multi_gpu) > 1:
            self.next_learning_rates = np.linspace(self.opt.lrate, self.opt.lrate * len(self.opt.multi_gpu),
                                                   5).tolist()
            self.next_learning_rates.reverse()
