import torch
import torch.optim as optim
import auxiliary.my_utils as my_utils
import json
import auxiliary.visualization as visualization
import os
import auxiliary.meter as meter
from termcolor import colored
import time


class AbstractTrainer(object):
    def __init__(self, opt):
        super(AbstractTrainer, self).__init__()
        self.start_time = time.time()
        self.opt = opt
        self.git_repo_path = ""
        self.start_visdom()
        self.get_log_paths()
        self.init_meters()
        self.reset_epoch()

        self.save_examples = False
        my_utils.print_arg(self.opt)

    def start_visdom(self):
        self.visualizer = visualization.Visualizer(self.opt.port, self.opt.env)

    def get_log_paths(self):
        """
        Get paths to save and reload networks
        :return:
        """
        if not os.path.exists("log"):
            print("Creating log folder")
            os.mkdir("log")
        if not os.path.exists(self.opt.dir_name):
            print("creating folder  ", self.opt.dir_name)
            os.mkdir(self.opt.dir_name)

        self.opt.logname = os.path.join(self.opt.dir_name, "log.txt")
        self.opt.checkpointname = os.path.join(self.opt.dir_name, 'optimizer_last.pth')

        # # If a network is already created in the directory
        self.opt.reload = False
        if os.path.exists(os.path.join(self.opt.dir_name, "network.pth")):
            print("Going to reload experiment")
            self.opt.reload = True

    def init_meters(self):
        self.log = meter.Logs()

    def print_loss_info(self):
        pass

    def build_optimizer(self):
        """
        Create optimizer
        """
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.opt.lrate)
        if self.opt.reload:
            self.optimizer.load_state_dict(torch.load(f'{self.opt.checkpointname}'))
            my_utils.yellow_print("Reloaded optimizer")

    def save_network(self):
        print("saving net...")
        torch.save(self.network.state_dict(), f"{self.opt.dir_name}/network.pth")
        torch.save(self.optimizer.state_dict(), f"{self.opt.dir_name}/optimizer_last.pth")
        print("network saved")

    def dump_stats(self):
        """
        Save stats at each epoch
        """

        log_table = {
            "epoch": self.epoch + 1,
            "lr": self.opt.lrate,
            "env": self.opt.env,
        }
        log_table.update(self.log.current_epoch)
        print(log_table)
        with open(self.opt.logname, "a") as f:  # open and append
            f.write("json_stats: " + json.dumps(log_table) + "\n")

        self.opt.start_epoch = self.epoch
        with open(os.path.join(self.opt.dir_name, "options.json"), "w") as f:  # open and append
            save_dict = dict(self.opt.__dict__)
            save_dict.pop("device")
            f.write(json.dumps(save_dict))

    def print_iteration_stats(self, loss):
        """
        print stats at each iteration
        """
        current_time = time.time()
        ellpased_time = current_time - self.start_train_time
        total_time_estimated = self.opt.nepoch * (self.datasets.len_dataset / self.opt.batch_size) * ellpased_time / (
                    0.00001 + self.iteration + 1.0 * self.epoch * self.datasets.len_dataset / self.opt.batch_size)  # regle de 3
        ETL = total_time_estimated - ellpased_time
        print(
            f"\r["
            + colored(f"{self.epoch}", "cyan")
            + f": "
            + colored(f"{self.iteration}", "red")
            + "/"
            + colored(f"{int(self.datasets.len_dataset / self.opt.batch_size)}", "red")
            + "] train loss:  "
            + colored(f"{loss.item()} ", "yellow")
            + colored(f"Ellapsed Time: {ellpased_time / 60 / 60}h ", "cyan")
            + colored(f"ETL: {ETL / 60 / 60}h", "red"),
            end="",
        )

    def learning_rate_scheduler(self):
        """
        Defines the learning rate schedule
        """
        # Warm-up following https://arxiv.org/pdf/1706.02677.pdf
        if len(self.next_learning_rates) > 0:
            next_learning_rate = self.next_learning_rates.pop()
            print(f"warm-up learning rate {next_learning_rate}")
            for g in self.optimizer.param_groups:
                g['lr'] = next_learning_rate


        if self.epoch == self.opt.lr_decay_1:
            self.opt.lrate = self.opt.lrate / 10.0
            print(f"First learning rate decay {self.opt.lrate}")
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.opt.lrate)
        if self.epoch == self.opt.lr_decay_2:
            self.opt.lrate = self.opt.lrate / 10.0
            print(f"Second learning rate decay {self.opt.lrate}")
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.opt.lrate)
        if self.epoch == self.opt.lr_decay_3:
            self.opt.lrate = self.opt.lrate / 10.0
            print(f"Third learning rate decay {self.opt.lrate}")
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.opt.lrate)

    def increment_epoch(self):
        self.epoch = self.epoch + 1

    def increment_iteration(self):
        self.iteration = self.iteration + 1

    def reset_iteration(self):
        self.iteration = 0

    def reset_epoch(self):
        self.epoch = self.opt.start_epoch

