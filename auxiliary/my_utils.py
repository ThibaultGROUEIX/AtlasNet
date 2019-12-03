import random
import numpy as np
import os
import torch
from termcolor import colored


def grey_print(x):
    print(colored(x, "grey"))


def red_print(x):
    print(colored(x, "red"))


def green_print(x):
    print(colored(x, "green"))


def yellow_print(x):
    print(colored(x, "yellow"))


def blue_print(x):
    print(colored(x, "blue"))


def magenta_print(x):
    print(colored(x, "magenta"))


def cyan_print(x):
    print(colored(x, "cyan"))


def white_print(x):
    print(colored(x, "white"))


def print_arg(opt):
    cyan_print("PARAMETER: ")
    for a in opt.__dict__:
        print(
            "         "
            + colored(a, "yellow")
            + " : "
            + colored(str(opt.__dict__[a]), "cyan")
        )

# initialize the weighs of the network for Convolutional layers and batchnorm layers
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def plant_seeds(randomized_seed=False):
    if randomized_seed:
        print("Randomized seed")
        manualSeed = random.randint(1, 10000)
    else:
        print("Used fix seed")
        manualSeed = 1
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)
