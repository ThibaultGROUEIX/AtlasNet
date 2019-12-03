import torch
import torch.optim as optim
import time
import auxiliary.my_utils as my_utils
import model
import dataset.dataset_shapenet as dataset_shapenet
import dataset.augmenter as augmenter
from training.abstract_trainer import AbstractTrainer
import os
import auxiliary.html_report as html_report
import numpy as np
from training.iteration import Iteration
from training.loss import Loss
from easydict import EasyDict
import model.model as model
import auxiliary.mesh_processor as mesh_processor
import pymesh

class Trainer(AbstractTrainer, Loss, Iteration):
    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.dataset_train = None
        self.opt.training_media_path = os.path.join(self.opt.dir_name, "training_media")
        if not os.path.exists(self.opt.training_media_path):
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

    def build_network(self):
        """
        Create network architecture. Refer to auxiliary.model
        :return:
        """
        self.opt.device = torch.device(f"cuda:{self.opt.multi_gpu[0]}")
        network = model.EncoderDecoder(self.opt)
        network.to(self.opt.device)

        if not self.opt.SVR:
            network.apply(my_utils.weights_init)  # initialization of the weight

        self.network = network
        self.network.eval()
        self.network.train()
        self.network = torch.nn.DataParallel(self.network, device_ids=self.opt.multi_gpu)
        if self.opt.model != "":
            try:
                self.network.load_state_dict(torch.load(self.opt.model))
                print(" Previous network weights loaded! From ", self.opt.model)
            except:
                print("Failed to reload ", self.opt.model)
        if self.opt.reload:
            print(f"reload model frow :  {self.opt.dir_name}/network.pth")
            self.opt.model = os.path.join(self.opt.dir_name, "network.pth")
            self.network.load_state_dict(torch.load(self.opt.model))

    def build_optimizer(self):
        """
        Create optimizer
        """
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.opt.lrate)
        if self.opt.reload:
            self.optimizer.load_state_dict(torch.load(f'{self.opt.checkpointname}'))
            my_utils.yellow_print("Reloaded optimizer")
        self.next_learning_rates = []
        if len(self.opt.multi_gpu) > 1:
            self.next_learning_rates = np.linspace(self.opt.lrate, self.opt.lrate * len(self.opt.multi_gpu),
                                                   5).tolist()
            self.next_learning_rates.reverse()

    def build_dataset(self):
        """
        Create dataset
        """
        self.datasets = EasyDict()
        self.datasets.dataset_train = dataset_shapenet.ShapeNet(self.opt, train=True)
        self.datasets.dataset_test = dataset_shapenet.ShapeNet(self.opt, train=False)

        if not self.opt.demo:
            self.datasets.dataloader_train = torch.utils.data.DataLoader(self.datasets.dataset_train,
                                                                         batch_size=self.opt.batch_size,
                                                                         shuffle=True, num_workers=int(self.opt.workers))
            self.datasets.dataloader_test = torch.utils.data.DataLoader(self.datasets.dataset_test,
                                                                        batch_size=self.opt.batch_size_test,
                                                                        shuffle=True, num_workers=int(self.opt.workers))
            axis = []
            if self.opt.data_augmentation_axis_rotation:
                axis = [1]

            flips = []
            if self.opt.data_augmentation_random_flips:
                flips = [0, 2]

            self.datasets.data_augmenter = augmenter.Augmenter(translation=self.opt.random_translation, rotation_axis=axis,
                                                               anisotropic_scaling=self.opt.anisotropic_scaling,
                                                               rotation_3D=self.opt.random_rotation,
                                                               flips=flips)
            self.datasets.len_dataset = len(self.datasets.dataset_train)
            self.datasets.len_dataset_test = len(self.datasets.dataset_test)

    def train_loop(self):
        iterator = self.datasets.dataloader_train.__iter__()
        for data in iterator:
            self.increment_iteration()
            # if self.iteration > 10:
            #     break
            self.data = EasyDict(data)
            self.data.points = self.data.points.to(self.opt.device)
            if self.datasets.data_augmenter is not None:
                self.datasets.data_augmenter(self.data.points)
            self.train_iteration()

    def train_epoch(self):
        start = time.time()
        self.flags.train = True

        if self.epoch == (self.opt.nepoch - 1):
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
        print("Ellapsed time : ", time.time() - start)

    def test_loop(self):
        iterator = self.datasets.dataloader_test.__iter__()
        self.reset_iteration()
        for data in iterator:
            self.increment_iteration()
            self.data = EasyDict(data)
            self.data.points = self.data.points.to(self.opt.device)
            self.test_iteration()

    def load_point_input(self, path):
        pass

    def load_image_input(self, path):
        pass

    def generate_random_mesh(self):
        index = np.random.randint(self.datasets.len_dataset_test)
        self.data = EasyDict(self.datasets.dataset_test[index])
        return self.generate_mesh()

    def generate_mesh(self):
        self.data.points.unsqueeze_(0)
        self.make_network_input()
        mesh = self.network.module.generate_mesh(self.data.network_input)
        path = '/'.join([self.opt.training_media_path, str(self.flags.media_count)]) + ".obj"
        image_path = '/'.join([self.data.image_path, '00.png'])
        mesh_processor.save(mesh, path, self.colormap)
        self.flags.media_count += 1
        return {"output_path": path,
                "image_path": image_path}

    def test_epoch(self):
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
        if (self.flags.build_website or self.opt.run_single_eval) and self.opt.compute_metro:
            self.metro()
        if self.flags.build_website:
            self.html_report_data = EasyDict()
            self.html_report_data.output_meshes = [self.generate_random_mesh() for i in range(3)]
            log_curves = ["loss_val", "loss_train_total"]
            self.html_report_data.data_curve = {key: [np.log(val) for val in self.log.curves[key]] for key in
                                                log_curves}
            self.html_report_data.fscore_curve = {"fscore": self.log.curves["fscore"]}
            html_report.main(self, outHtml="index.html")

    def demo(self, demo_path):
        """
        This function takes an image or pointcloud path as input and save the mesh infered by Atlasnet
        Extension supported are ply npy obg and png
        :return: path to the generated mesh
        """
        ext = demo_path.split('.')[-1]
        if ext == "ply" or ext == "npy" or ext == "obj":
            self.data = self.datasets.dataset_train.load_point_input(demo_path)
        elif ext == "png":
            self.data = self.datasets.dataset_train.load_image(demo_path)
        else:
            print("invalid file extension")

        self.data = EasyDict(self.data)
        self.make_network_input()
        mesh = self.network.module.generate_mesh(self.data.network_input)
        if self.data.operation is not None:
            vertices = torch.from_numpy(mesh.vertices).clone().unsqueeze(0)
            self.data.operation.invert()
            unnormalized_vertices = self.data.operation.apply(vertices)
            mesh = pymesh.form_mesh(vertices=unnormalized_vertices.squeeze().numpy(), faces=mesh.faces)

        if self.opt.demo:
            path = demo_path.split('.')
            path[-2] += "AtlasnetReconstruction"
            path[-1] = "ply"
            path = ".".join(path)
        else:
            path = '/'.join([self.opt.training_media_path, str(self.flags.media_count)]) + ".ply"
            self.flags.media_count += 1

        print(f"Atlasnet saved generated mesh at {path}!")
        mesh_processor.save(mesh, path, self.colormap)
        return path