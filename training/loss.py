import torch
from easydict import EasyDict
import ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as dist_chamfer_3D


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
        return
