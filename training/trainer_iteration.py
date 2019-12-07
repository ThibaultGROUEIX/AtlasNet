from termcolor import colored


class TrainerIteration(object):
    """
        This class implements all functions related to a single forward pass of Atlasnet.
        Author : Thibault Groueix 01.11.2019
    """

    def __init__(self):
        super(TrainerIteration, self).__init__()

    def make_network_input(self):
        """
        Arrange to data to be fed to the network.
        :return:
        """
        if self.opt.SVR:
            self.data.network_input = self.data.image.to(self.opt.device)
        else:
            self.data.network_input = self.data.points.transpose(2, 1).contiguous().to(self.opt.device)

    def common_ops(self):
        """
        Commom operations between train and eval forward passes
        :return:
        """
        self.make_network_input()
        self.batch_size = self.data.points.size(0)

        self.data.pointsReconstructed_prims = self.network(self.data.network_input,
                                                           train=self.flags.train)
        self.fuse_primitives()

        self.loss_model()  # batch
        self.visualize()

    def train_iteration(self):
        """
        Forward backward pass
        :return:
        """
        self.optimizer.zero_grad()
        self.common_ops()
        self.log.update("loss_train_total", self.data.loss.item())
        if not self.opt.no_learning:
            self.data.loss.backward()
            self.optimizer.step()  # gradient update
        self.print_iteration_stats(self.data.loss)

    def visualize(self):
        if self.iteration % 50 == 1:
            tmp_string = "train" if self.flags.train else "test"
            self.visualizer.show_pointcloud(self.data.points[0], title=f"GT {tmp_string}")
            self.visualizer.show_pointcloud(self.data.pointsReconstructed[0], title=f"Reconstruction {tmp_string}")
            if self.opt.SVR:
                self.visualizer.show_image(self.data.image[0], title=f"Input Image {tmp_string}")

    def test_iteration(self):
        """
        Forward evaluation pass
        :return:
        """
        self.common_ops()
        self.num_val_points = self.data.pointsReconstructed.size(1)
        self.log.update("loss_val", self.data.loss.item())
        self.log.update("fscore", self.data.loss_fscore.item())
        print(
            '\r' + colored(
                '[%d: %d/%d]' % (self.epoch, self.iteration, self.datasets.len_dataset_test / self.opt.batch_size_test),
                'red') +
            colored('loss_val:  %f' % self.data.loss.item(), 'yellow'),
            end='')
