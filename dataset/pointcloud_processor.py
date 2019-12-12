import numpy as np
import torch


"""
Author : Thibault Groueix 01.09.2019

Standards :
-- For batch processing. If you have a 2D tensor, it will unsqueeze it first
-- Support in place and out of place
-- Support inversion of applied transformation
"""


class FunctionGenerator(object):
    def invert(self):
        print("This function has to be reimplemented in every inherited class")


class ScaleFunctions(FunctionGenerator):
    def __init__(self, operator, inplace):
        self.operator = operator.clone()
        self.inplace = inplace

    def __call__(self, points):
        if self.inplace:
            points *= self.operator
            return points
        else:
            return points * self.operator

    def invert(self):
        self.operator = 1.0 / self.operator


class RotationFunctions(FunctionGenerator):
    def __init__(self, operator, inplace):
        self.operator = operator.clone()
        self.inplace = inplace
        assert (self.operator.bmm(self.operator.transpose(1, 2).contiguous()).sum().item() - (
                operator.size(0) * operator.size(2))) ** 2 < 0.001, "Input matrix is not a rotation matrix"

    def __call__(self, points):
        rotated_points = torch.bmm(points, self.operator)
        if self.inplace:
            points.copy_(rotated_points)
            return points
        return rotated_points

    def invert(self):
        self.operator = self.operator.transpose(1, 2).contiguous()


class TranslationFunctions(FunctionGenerator):
    def __init__(self, operator, inplace):
        self.operator = operator.clone()
        self.inplace = inplace

    def __call__(self, points):
        if self.inplace:
            points += self.operator
            return points
        else:
            return points + self.operator

    def invert(self):
        self.operator = -self.operator


class Operation(object):
    def __init__(self, points, inplace=True, keep_track=False):
        """
        The keep track boolean is used in case one wants to unroll all the operation that have been performed
        :param keep_track: boolean
        """
        self.keep_track = keep_track
        self.transforms = []
        self.points = points
        self.device = points.device
        self.inplace = inplace
        self.dim = points.dim()
        self.type = self.points.type()

        if not self.inplace:
            self.points = self.points.clone()
        if self.dim == 2:
            self.points = self.points.unsqueeze_(0)
        elif self.dim == 3:
            pass
        else:
            print("Input should have dimension 2 or 3")

    def apply(self, points):
        for func in self.transforms:
            points = func(points)
        return points

    def invert(self):
        self.transforms.reverse()
        for func in self.transforms:
            func.invert()

    def scale(self, scale_vector):
        scaling_op = ScaleFunctions(scale_vector.to(self.device).type(self.type), inplace=self.inplace)
        self.points = scaling_op(self.points)
        if self.keep_track:
            self.transforms.append(scaling_op)
        return

    def translate(self, translation_vector):
        translation_op = TranslationFunctions(translation_vector.to(self.device).type(self.type), inplace=self.inplace)
        self.points = translation_op(self.points)
        if self.keep_track:
            self.transforms.append(translation_op)
        return

    def rotate(self, rotation_vector):
        rotation_op = RotationFunctions(rotation_vector.to(self.device).type(self.type), inplace=self.inplace)
        self.points = rotation_op(self.points)
        if self.keep_track:
            self.transforms.append(rotation_op)
        return

    @staticmethod
    def get_3D_rot_matrix(axis, rad_angle):
        """
        Get a 3D rotation matrix around axis with angle in radian
        :param axis: int
        :param angle: torch.tensor of size Batch.
        :return: Rotation Matrix as a tensor
        """
        cos_angle = torch.cos(rad_angle)
        sin_angle = torch.sin(rad_angle)
        rotation_matrix = torch.zeros(rad_angle.size(0), 3, 3)
        rotation_matrix[:, 1, 1].fill_(1)
        rotation_matrix[:, 0, 0].copy_(cos_angle)
        rotation_matrix[:, 0, 2].copy_(sin_angle)
        rotation_matrix[:, 2, 0].copy_(-sin_angle)
        rotation_matrix[:, 2, 2].copy_(cos_angle)
        if axis == 0:
            rotation_matrix = rotation_matrix[:, [1, 0, 2], :][:, :, [1, 0, 2]]
        if axis == 2:
            rotation_matrix = rotation_matrix[:, [0, 2, 1], :][:, :, [0, 2, 1]]
        return rotation_matrix

    def rotate_axis_angle(self, axis, rad_angle, normals=False):
        """

        :param points: Batched points
        :param axis: int
        :param angle: batched angles
        :return:
        """
        rot_matrix = Operation.get_3D_rot_matrix(axis=axis, rad_angle=rad_angle)
        if normals:
            rot_matrix = torch.cat([rot_matrix, rot_matrix], dim=2)
        self.rotate(rot_matrix)
        return


class Normalization(Operation):
    def __init__(self, *args, **kwargs):
        super(Normalization, self).__init__(*args, **kwargs)

    def center_pointcloud(self):
        """
        In-place centering
        :param points:  Tensor Batch, N_pts, D_dim
        :return: None
        """
        # input :
        # ouput : torch Tensor N_pts, D_dim
        centroid = torch.mean(self.points, dim=1, keepdim=True)
        self.translate(-centroid)
        return self.points

    @staticmethod
    def center_pointcloud_functional(points):
        operator = Normalization(points, inplace=False)
        return operator.center_pointcloud()

    def normalize_unitL2ball(self):
        """
        In-place normalization of input to unit ball
        :param points: torch Tensor Batch, N_pts, D_dim
        :return: None
        """
        # input : torch Tensor N_pts, D_dim
        # ouput : torch Tensor N_pts, D_dim
        #
        self.center_pointcloud()
        scaling_factor_square, _ = torch.max(torch.sum(self.points ** 2, dim=2, keepdim=True), dim=1, keepdim=True)
        scaling_factor = torch.sqrt(scaling_factor_square)
        self.scale(1.0 / scaling_factor)
        return self.points

    @staticmethod
    def normalize_unitL2ball_functional(points):
        operator = Normalization(points, inplace=False)
        return operator.normalize_unitL2ball()

    def center_bounding_box(self):
        """
        in place Centering : return center the bounding box
        :param points: torch Tensor Batch, N_pts, D_dim
        :return: diameter
        """
        min_vals, _ = torch.min(self.points, 1, keepdim=True)
        max_vals, _ = torch.max(self.points, 1, keepdim=True)
        self.translate(-(min_vals + max_vals) / 2)
        return self.points, (max_vals - min_vals) / 2

    @staticmethod
    def center_bounding_box_functional(points):
        operator = Normalization(points, inplace=False)
        points, _ = operator.center_bounding_box()
        return points

    def normalize_bounding_box(self, isotropic=True):
        """
        In place : center the bounding box and uniformly scale the bounding box to edge lenght 1 or max edge length 1 if isotropic is True  (default).
        :param points: torch Tensor Batch, N_pts, D_dim
        :return:
        """
        _, diameter = self.center_bounding_box()
        if isotropic:
            diameter, _ = torch.max(diameter, 2, keepdim=True)
        self.scale(1.0 / diameter)
        return self.points

    @staticmethod
    def normalize_bounding_box_functional(points):
        operator = Normalization(points, inplace=False)
        return operator.normalize_bounding_box()

    @staticmethod
    def identity_functional(points):
        return points


class DataAugmentation(Operation):
    def __init__(self, *args, **kwargs):
        super(DataAugmentation, self).__init__(*args, **kwargs)

    def random_anisotropic_scaling(self, min_val=0.75, max_val=1.25):
        """
        In place : Random Anisotropic scaling by a factor between min_val and max_val
        :param points: torch Tensor Batch, N_pts, D_dim
        :return:
        """
        scale = torch.rand(self.points.size(0), 1, self.points.size(2)) * (max_val - min_val) + min_val
        self.scale(scale)
        return

    def random_axial_rotation(self, axis=0, normals=False, range_rot=360):
        """
        Compute a random rotation of the batch around an axis. There is is no in-place version of this function because bmm_ is not possible in pytorch.
        :param points: torch Tensor Batch, N_pts, D_dim
        :return: torch Tensor Batch, N_pts, D_dim
        """
        scale_factor = 360.0 / range_rot
        scale_factor = np.pi / scale_factor
        rad_angle = torch.rand(self.points.size(0)) * 2 * scale_factor - scale_factor
        self.rotate_axis_angle(axis=axis, rad_angle=rad_angle, normals=normals)
        return

    @staticmethod
    def get_random_rotation_matrix(batch_size=1):
        """
        Get a random 3D rotation matrix
        :return: Rotation Matrix as a tensor
        from : https://en.wikipedia.org/wiki/Rotation_matrix#Basic_rotations
        An easy way to do this : sample a point on the sphere (with normalize(normal(), normal(), normal())
         then sample an angle, then just compute the associated rotation matrix
        """
        # Select a random point on the sphere
        x = torch.randn(batch_size, 1, 3).double()
        scaling_factor_square, _ = torch.max(torch.sum(x ** 2, dim=2, keepdim=True), dim=1, keepdim=True)
        scaling_factor = torch.sqrt(scaling_factor_square)
        x /= scaling_factor
        x = x.squeeze()
        XX = torch.bmm(x.unsqueeze(2), x.unsqueeze(1))

        # get random angle
        rad_angle = torch.rand(batch_size).double() * 2 * np.pi + np.pi
        cos_angle = torch.cos(rad_angle)
        sin_angle = torch.sin(rad_angle)

        # Compute fat matrix
        rotation_matrix = torch.zeros(rad_angle.size(0), 3, 3).double()

        rotation_matrix[:, 0, 0].copy_(cos_angle + XX[:, 0, 0] * (1 - cos_angle))
        rotation_matrix[:, 1, 1].copy_(cos_angle + XX[:, 1, 1] * (1 - cos_angle))
        rotation_matrix[:, 2, 2].copy_(cos_angle + XX[:, 2, 2] * (1 - cos_angle))

        rotation_matrix[:, 0, 1].copy_(XX[:, 0, 1] * (1 - cos_angle) - x[:, 2] * sin_angle)
        rotation_matrix[:, 1, 0].copy_(XX[:, 0, 1] * (1 - cos_angle) + x[:, 2] * sin_angle)

        rotation_matrix[:, 0, 2].copy_(XX[:, 0, 2] * (1 - cos_angle) + x[:, 1] * sin_angle)
        rotation_matrix[:, 2, 0].copy_(XX[:, 0, 2] * (1 - cos_angle) - x[:, 1] * sin_angle)

        rotation_matrix[:, 1, 2].copy_(XX[:, 1, 2] * (1 - cos_angle) - x[:, 0] * sin_angle)
        rotation_matrix[:, 2, 1].copy_(XX[:, 1, 2] * (1 - cos_angle) + x[:, 0] * sin_angle)

        return rotation_matrix

    def random_rotation(self, normals=False):
        """
        Compute a random rotation of the batch. There is is no in-place version of this function because bmm_ is not possible in pytorch.
        :param points: torch Tensor Batch, N_pts, D_dim
        :return: torch Tensor Batch, N_pts, D_dim
        """
        rot_matrix = DataAugmentation.get_random_rotation_matrix(batch_size=self.points.size(0))
        if normals:
            rot_matrix = torch.cat([rot_matrix, rot_matrix], dim=2)
        self.rotate(rot_matrix)
        return

    def random_translation(self, scale=0.03):
        """
        In place Compute a random tranlation of the batch.
        :param points: torch Tensor Batch, N_pts, D_dim
        :return:
        """
        translation_vector = torch.rand(self.points.size(0), 1, self.points.size(2)) * 2 * scale - scale
        self.translate(translation_vector)
        return

    @staticmethod
    def diff(first, second):
        second = set(second)
        return [item for item in first if item not in second]

    def random_flips(self, dims=[]):
        """
               In place Random flip
               :param points: torch Tensor Batch, N_pts, D_dim
               :return:
        """
        exclude_dims = DataAugmentation.diff(range(self.points.size(2)), dims)
        scale_factor = torch.randint(2, (self.points.size(0), 1, self.points.size(2))) * 2 - 1
        for axis in exclude_dims:
            scale_factor[:, :, axis].fill_(1)
        self.scale(scale_factor)
        return


# Done for eurographics 19
def barycentric(p, a, b, c):
    """
    :param p: numpy arrays of size N_points x 3
    :param a: numpy arrays of size N_points x 3
    :param b: numpy arrays of size N_points x 3
    :param c: numpy arrays of size N_points x 3
    :return: barycentric coordinates point p in triangle (a,b,c)
    """

    v0 = b - a
    v1 = c - a
    v2 = p - a

    d00 = np.sum(np.multiply(v0, v0), 1)
    d01 = np.sum(np.multiply(v0, v1), 1)
    d11 = np.sum(np.multiply(v1, v1), 1)
    d20 = np.sum(np.multiply(v2, v0), 1)
    d21 = np.sum(np.multiply(v2, v1), 1)

    denom = np.multiply(d00, d11) - np.multiply(d01, d01)

    v = (np.multiply(d11, d20) - np.multiply(d01, d21)) / denom
    w = (np.multiply(d00, d21) - np.multiply(d01, d20)) / denom
    u = - v - w + 1.0

    return (u, v, w)


if __name__ == '__main__':
    print("Start unit test")
