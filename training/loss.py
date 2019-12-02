import torch
from easydict import EasyDict
import auxiliary.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as dist_chamfer_3D
from auxiliary.ChamferDistancePytorch.fscore import fscore
import os
import auxiliary.mesh_processor as mesh_processor
import training.metro as metro
import pymesh
from joblib import Parallel, delayed
import numpy as np

class Loss(object):
    def __init__(self):
        super(Loss, self).__init__()

    def build_losses(self):
        """
        Create losses
        """
        self.distChamfer = dist_chamfer_3D.chamfer_3DDist()
        self.loss_model = self.chamfer_loss_union

    def fuse_primitives(self):
        # prim, batch, exp, 2, npoints
        self.data.pointsReconstructed = self.data.pointsReconstructed_prims.transpose(2,
                                                                                      3).contiguous()  # batch, exp, prim, npoints, 2,
        self.data.pointsReconstructed = self.data.pointsReconstructed.view(self.batch_size, -1, 3)

    def chamfer_loss_union(self):
        inCham1 = self.data.points.view(self.data.points.size(0), self.opt.number_points, 3).contiguous()
        inCham2 = self.data.pointsReconstructed.contiguous().view(self.data.points.size(0), -1, 3).contiguous()

        dist1, dist2, idx1, idx2 = self.distChamfer(inCham1, inCham2)  # mean over points
        self.data.loss = torch.mean(dist1) + torch.mean(dist2)  # mean over points
        if not self.flags.train:
            self.data.loss_fscore, _, _ = fscore(dist1, dist2)
            self.data.loss_fscore = self.data.loss_fscore.mean()

    def metro(self):
        metro_path = './dataset/data/metro_files'
        metro_files_path = '/'.join([metro_path, 'files-metro.txt'])
        self.metro_args_input = []
        if not os.path.exists(metro_files_path):
            os.system("chmod +x dataset/download_metro_files.sh")
            os.system("./dataset/download_metro_files.sh")

        with open(metro_files_path, 'r') as file:
            files = file.read().split('\n')
        for file in files:
            if file[-3:] == "ply":
                cat = file.split('/')[0]
                name = file.split('/')[1][:-4]
                pointcloud_path = '/'.join([metro_path, cat, name + '.npy'])
                gt_path = '/'.join([metro_path, cat, name + '.ply'])
                self.data = self.datasets.dataset_train.load_point_input(pointcloud_path)
                self.data = EasyDict(self.data)
                self.make_network_input()
                mesh = self.network.module.generate_mesh(self.data.network_input)
                vertices = torch.from_numpy(mesh.vertices).clone().unsqueeze(0)
                self.data.operation.invert()
                unnormalized_vertices = self.data.operation.apply(vertices)
                mesh = pymesh.form_mesh(vertices=unnormalized_vertices.squeeze().numpy(), faces=mesh.faces)
                path = '/'.join([self.opt.training_media_path, str(self.flags.media_count)]) + ".ply"
                mesh_processor.save(mesh, path, self.colormap)
                self.flags.media_count += 1
                self.metro_args_input.append((path, gt_path))


        print("start metro calculus. This is going to take some time (30 minutes)")
        self.metro_results = Parallel(n_jobs=-1, backend="multiprocessing")(
            delayed(metro.metro)(*i) for i in self.metro_args_input)
        self.metro_results = np.array(self.metro_results).mean()
