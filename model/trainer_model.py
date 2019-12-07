import torch
import model.model as model
import auxiliary.my_utils as my_utils
import model
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
        self.opt.device = torch.device(f"cuda:{self.opt.multi_gpu[0]}")
        network = model.EncoderDecoder(self.opt)
        self.reload_network()
        self.network = nn.DataParallel(network, device_ids=self.opt.multi_gpu)

    def reload_network(self):
        """
        Reload entire model or only decoder (atlasnet) depending on the options
        :return:
        """
        if self.opt.reload_model_path != "":
            try:
                self.network.load_state_dict(torch.load(self.opt.model))
                my_utils.yellow_print(f"Network weights loaded from  {self.opt.model}!")
            except:
                my_utils.yellow_print(f"Failed to reload {self.opt.model}")

        elif self.opt.reload_decoder_path != "":
            try:
                self.network.decoder.load_state_dict(torch.load(self.opt.reload_decoder_path))
                my_utils.yellow_print(f"Network weights loaded from  {self.opt.reload_decoder_path}!")
            except:
                my_utils.yellow_print(f"Failed to reload {self.opt.reload_decoder_path}")
        else:
            my_utils.yellow_print("No network weights to reload!")

    def build_optimizer(self):
        """
        Create optimizer
        """
        if self.opt.train_only_encoder:
            # To train a resnet image encoder with a pre-trained atlasnet decoder.
            self.optimizer = optim.Adam(self.network.modules.encoder.parameters(), lr=self.opt.lrate)
        else:
            self.optimizer = optim.Adam(self.network.modules.parameters(), lr=self.opt.lrate)

        if self.opt.reload_optimizer_path != "":
            try:
                self.optimizer.load_state_dict(torch.load(self.opt.reload_optimizer_path))
                my_utils.yellow_print(f"Reloaded optimizer {self.opt.reload_optimizer_path}")
            except:
                my_utils.yellow_print(f"Failed to reload optimizer {self.opt.reload_optimizer_path}")

        # Set policy for warm-up if you use multiple GPUs
        self.next_learning_rates = []
        if len(self.opt.multi_gpu) > 1:
            self.next_learning_rates = np.linspace(self.opt.lrate, self.opt.lrate * len(self.opt.multi_gpu),
                                                   5).tolist()
            self.next_learning_rates.reverse()
