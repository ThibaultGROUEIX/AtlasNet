import random
import numpy as np
import os
import torch
from termcolor import colored
from mpl_toolkits.mplot3d import Axes3D
Axes3D = Axes3D  # pycharm auto import

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


def int_2_boolean(x):
    if x == 1:
        return True
    else:
        return False



#initialize the weighs of the network for Convolutional layers and batchnorm layers
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def adjust_learning_rate(optimizer, epoch, phase):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if (epoch%phase==(phase-1)):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']/10.


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


class AverageValueMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_random_color(pastel_factor = 0.5):
    return [(x+pastel_factor)/(1.0+pastel_factor) for x in [random.uniform(0,1.0) for i in [1,2,3]]]

def color_distance(c1,c2):
    return sum([abs(x[0]-x[1]) for x in zip(c1,c2)])

def generate_new_color(existing_colors,pastel_factor = 0.5):
    max_distance = None
    best_color = None
    for i in range(0,100):
        color = get_random_color(pastel_factor = pastel_factor)
        if not existing_colors:
            return color
        best_distance = min([color_distance(color,c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color

#Example:
def get_colors(num_colors=10):
  colors = []
  for i in range(0,num_colors):
      colors.append(generate_new_color(colors,pastel_factor = 0.9))
  for i in range(0,num_colors):
      for j in range(0,3):
        colors[i][j] = int(colors[i][j]*256)
      colors[i].append(255)
  return colors


CHUNK_SIZE = 150
lenght_line = 60
def my_get_n_random_lines(path, n=5):
    MY_CHUNK_SIZE = lenght_line * (n+2)
    lenght = os.stat(path).st_size
    with open(path, 'r') as file:
            file.seek(random.randint(400, lenght - MY_CHUNK_SIZE))
            chunk = file.read(MY_CHUNK_SIZE)
            lines = chunk.split(os.linesep)
            return lines[1:n+1]

def sampleSphere(N):

    rand = np.random.rand(N)

    theta = 2*np.pi*np.random.rand(N)
    phi = np.arccos(1 - 2*np.random.rand(N))
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    sphere = np.array([x,y,z]).transpose(1,0)

    return sphere

if __name__ == '__main__':

  #To make your color choice reproducible, uncomment the following line:
  #random.seed(10)

  sampleSphere(1000)
