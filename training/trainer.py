import torch
import os
import auxiliary.html_report as html_report
import numpy as np
from easydict import EasyDict
import pymesh

from training.trainer_abstract import TrainerAbstract
import dataset.mesh_processor as mesh_processor
from training.trainer_iteration import TrainerIteration
from model.trainer_model import TrainerModel
from dataset.trainer_dataset import TrainerDataset
from training.trainer_loss import TrainerLoss


class Trainer(TrainerAbstract, TrainerLoss, TrainerIteration, TrainerDataset, TrainerModel):
    def __init__(self, opt):
        """
        Main Atlasnet class inheriting from the other main modules.
        It implements all functions related to train and evaluate for an epoch.
        Author : Thibault Groueix 01.11.2019
        :param opt:
        """

        super(Trainer, self).__init__(opt)
        self.dataset_train = None
        self.opt.training_media_path = os.path.join(self.opt.dir_name, "training_media")
        if not opt.demo and not os.path.exists(self.opt.training_media_path):
            os.mkdir(self.opt.training_media_path)

        # Define Flags
        self.flags = EasyDict()
        self.flags.media_count = 0
        self.flags.add_log = True
        self.flags.build_website = False
        self.flags.get_closer_neighbourg = False
        self.flags.compute_clustering_errors = False
        self.display = EasyDict({"recons": []})
        self.colormap = mesh_processor.ColorMap()

    def train_loop(self):
        """
        Take a single pass on all train data
        :return:
        """
        iterator = self.datasets.dataloader_train.__iter__()
        for data in iterator:
            self.increment_iteration()
            self.data = EasyDict(data)
            self.data.points = self.data.points.to(self.opt.device)
            if self.datasets.data_augmenter is not None and not self.opt.SVR:
                # Apply data augmentation on points
                self.datasets.data_augmenter(self.data.points)

            self.train_iteration()

    def train_epoch(self):
        """ Launch an train epoch """
        self.flags.train = True
        if self.epoch == (self.opt.nepoch - 1):
            # Flag last epoch
            self.flags.build_website = True

        self.log.reset()
        if not self.opt.no_learning:
            self.network.train()
        else:
            self.network.eval()
        self.learning_rate_scheduler()
        self.reset_iteration()
        for i in range(self.opt.loop_per_epoch):
            self.train_loop()

    def test_loop(self):
        """
        Take a single pass on all test data
        :return:
        """
        iterator = self.datasets.dataloader_test.__iter__()
        self.reset_iteration()
        for data in iterator:
            self.increment_iteration()
            self.data = EasyDict(data)
            self.data.points = self.data.points.to(self.opt.device)
            self.test_iteration()

    def test_epoch(self):
        """ Launch an test epoch """
        self.flags.train = False
        self.network.eval()
        self.test_loop()
        self.log.end_epoch()

        try:
            self.log.update_curves(self.visualizer.vis, self.opt.dir_name)
        except:
            print("could not update curves")
        print(f"Sampled {self.num_val_points} regular points for evaluation")

        self.metro_results = 0
        if (self.flags.build_website or self.opt.run_single_eval) and not self.opt.no_metro:
            self.metro()

        if self.flags.build_website:
            # Build report using Netvision.
            self.html_report_data = EasyDict()
            self.html_report_data.output_meshes = [self.generate_random_mesh() for i in range(3)]
            log_curves = ["loss_val", "loss_train_total"]
            self.html_report_data.data_curve = {key: [np.log(val) for val in self.log.curves[key]] for key in
                                                log_curves}
            self.html_report_data.fscore_curve = {"fscore": self.log.curves["fscore"]}
            html_report.main(self, outHtml="index.html")

    def generate_random_mesh(self):
        """ Generate a mesh from a random test sample """
        index = np.random.randint(self.datasets.len_dataset_test)
        self.data = EasyDict(self.datasets.dataset_test[index])
        self.data.points.unsqueeze_(0)
        if self.opt.SVR:
            self.data.image.unsqueeze_(0)
        return self.generate_mesh()

    def generate_mesh(self):
        """
        Generate a mesh from self.data and saves it.
        :return:
        """
        self.make_network_input()
        mesh = self.network.module.generate_mesh(self.data.network_input)
        path = '/'.join([self.opt.training_media_path, str(self.flags.media_count)]) + ".obj"
        image_path = '/'.join([self.data.image_path, '00.png'])
        mesh_processor.save(mesh, path, self.colormap)
        self.flags.media_count += 1
        return {"output_path": path,
                "image_path": image_path}

    def demo(self, demo_path, input_path_points=None):
        """
        This function takes an image or pointcloud path as input and save the mesh infered by Atlasnet
        Extension supported are ply npy obg and png
        :return: path to the generated mesh
        """
        ext = demo_path.split('.')[-1]
        self.data = self.datasets.dataset_train.load(demo_path)
        self.data = EasyDict(self.data)

        if input_path_points is None:
            input_path_points = demo_path

        #prepare normalization
        get_normalization = self.datasets.dataset_train.load(input_path_points)
        get_normalization = EasyDict(get_normalization)

        self.make_network_input()
        mesh = self.network.module.generate_mesh(self.data.network_input)
        if get_normalization.operation is not None:
            # Undo any normalization that was used to preprocess the input.
            vertices = torch.from_numpy(mesh.vertices).clone().unsqueeze(0)
            get_normalization.operation.invert()
            unnormalized_vertices = get_normalization.operation.apply(vertices)
            mesh = pymesh.form_mesh(vertices=unnormalized_vertices.squeeze().numpy(), faces=mesh.faces)

        if self.opt.demo:
            path = demo_path.split('.')
            path[-2] += "AtlasnetReconstruction"
            path[-1] = "ply"
            path = ".".join(path)
        else:
            path = '/'.join([self.opt.training_media_path, str(self.flags.media_count)]) + ".ply"
            self.flags.media_count += 1

        print(f"Atlasnet generated mesh at {path}!")
        mesh_processor.save(mesh, path, self.colormap)
        return path
