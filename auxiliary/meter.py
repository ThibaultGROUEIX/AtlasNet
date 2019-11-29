import numpy as np
from termcolor import colored
import matplotlib.pyplot as plt
import os
import torch


class AverageValueMeter(object):
    """
    Slightly fancier than the standard AverageValueMeter
    """

    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0

    def update(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.avg, self.std = np.nan, np.nan
        elif self.n == 1:
            self.avg = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.avg_old = self.avg
            self.m_s = 0.0
        else:
            self.avg = self.avg_old + (value - n * self.avg_old) / float(self.n)
            self.m_s += (value - self.avg_old) * (value - self.avg)
            self.avg_old = self.avg
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        return self.avg, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.avg = 0
        self.avg_old = 0.0
        self.m_s = 0.0
        self.std = np.nan


def convert_label_to_one_hot_numpy(labels, num_categories):
    labels = labels.view(len(labels), 1)
    label_one_hot = np.zeros((labels.shape[0], num_categories))
    for idx in range(labels.shape[0]):
        label_one_hot[idx, labels[idx]] = 1
    return label_one_hot


def convert_label_to_one_hot_torch(labels, num_categories):
    labels = labels.view(labels.nelement(), 1).long()
    labels_onehot = torch.FloatTensor(len(labels), num_categories).to(labels.device)
    labels_onehot.zero_()
    labels_onehot.scatter_(1, labels, 1)
    return labels_onehot


def convert_one_hot_to_labels(one_hot):
    return torch.max(one_hot, 1)[1]


class Logs(object):
    def __init__(self, curves=[]):
        self.curves_names = curves
        self.curves = {}
        self.meters = {}
        self.current_epoch = {}
        for name in self.curves_names:
            self.curves[name] = []
            self.meters[name] = AverageValueMeter()

    def end_epoch(self):
        """
        Add meters average in average list and keep in current_epoch the current statistics
        :return:
        """
        for name in self.curves_names:
            self.curves[name].append(self.meters[name].avg)
            if len(name) < 20:
                print(colored(name, 'yellow') + " " + colored(f"{self.meters[name].avg}", 'cyan'))

            self.current_epoch[name] = self.meters[name].avg

    def reset(self):
        """
        Reset all meters
        :return:
        """
        for name in self.curves_names:
            self.meters[name].reset()

    def update(self, name, val):
        """
        :param name: meter name
        :param val: new value to add
        :return: void
        """
        if not name in self.curves_names:
            print(f"adding {name} to the log curves to display")
            self.meters[name] = AverageValueMeter()
            self.curves[name] = []
            self.curves_names.append(name)
        try:
            self.meters[name].update(val.item())
        except:
            self.meters[name].update(val)

    @staticmethod
    def stack_numpy_array(A, B):
        if A is None:
            return B
        else:
            return np.column_stack((A, B))

    def plot_bar(self, vis, name):
        vis.bar(self.meters[name].avg, win='barplot')  # Here

    def update_curves(self, vis, path):
        X_Loss = None
        Y_Loss = None
        Names_Loss = []


        for name in self.curves_names:
            if name[:4] == "loss":
                Names_Loss.append(name)
                X_Loss = self.stack_numpy_array(X_Loss, np.arange(len(self.curves[name])))
                Y_Loss = self.stack_numpy_array(Y_Loss, np.array(self.curves[name]))

            else:
                print(f"Dont know what to do with {name}")

        vis.line(X=X_Loss,
                 Y=Y_Loss,
                 win='loss',
                 opts=dict(title="loss", legend=Names_Loss))

        vis.line(X=X_Loss,
                 Y=np.log(Y_Loss),
                 win='log',
                 opts=dict(title="log", legend=Names_Loss))
        try:
            vis.line(X=np.arange(len(self.curves["fscore"])),
                     Y=self.curves["fscore"],
                     win='fscore',
                     opts=dict(title="fscore"))
        except:
            pass
        # Save figures in PNGs
        plt.figure()
        for i in range(X_Loss.shape[1]):
            plt.plot(X_Loss[:, i], Y_Loss[:, i], label=Names_Loss[i])
        plt.title('Curves')
        plt.legend()
        plt.savefig(os.path.join(path, "curve.png"))

        plt.figure()
        for i in range(X_Loss.shape[1]):
            plt.plot(X_Loss[:, i], np.log(Y_Loss[:, i]), label=Names_Loss[i])
        plt.title('Curves in Log')
        plt.legend()
        plt.savefig(os.path.join(path, "curve_log.png"))
