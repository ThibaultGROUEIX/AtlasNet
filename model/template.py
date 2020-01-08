import pymesh
import numpy as np
import torch
from torch.autograd import Variable

"""
        Author : Thibault Groueix 01.11.2019
"""


def get_template(template_type, device=0):
    getter = {
        "SQUARE": SquareTemplate,
        "SPHERE": SphereTemplate,
    }
    template = getter.get(template_type, "Invalid template")
    return template(device=device)


class Template(object):
    def get_random_points(self):
        print("Please implement get_random_points ")

    def get_regular_points(self):
        print("Please implement get_regular_points ")


class SphereTemplate(Template):
    def __init__(self, device=0, grain=6):
        self.device = device
        self.dim = 3
        self.npoints = 0

    def get_random_points(self, shape, device="gpu0"):
        """
        Get random points on a Sphere
        Return Tensor of Size [x, 3, x ... x]
        """
        assert shape[1] == 3, "shape should have 3 in dim 1"
        rand_grid = torch.cuda.FloatTensor(shape).to(device).float()
        rand_grid.data.normal_(0, 1)
        rand_grid = rand_grid / torch.sqrt(torch.sum(rand_grid ** 2, dim=1, keepdim=True))
        return Variable(rand_grid)

    def get_regular_points(self, npoints=None, device="gpu0"):
        """
        Get regular points on a Sphere
        Return Tensor of Size [x, 3]
        """
        if not self.npoints == npoints:
            self.mesh = pymesh.generate_icosphere(1, [0, 0, 0], 4)  # 2562 vertices
            self.vertex = torch.from_numpy(self.mesh.vertices).to(device).float()
            self.num_vertex = self.vertex.size(0)
            self.vertex = self.vertex.transpose(0,1).contiguous().unsqueeze(0)
            self.npoints = npoints

        return Variable(self.vertex.to(device))


class SquareTemplate(Template):
    def __init__(self, device=0):
        self.device = device
        self.dim = 2
        self.npoints = 0

    def get_random_points(self, shape, device="gpu0"):
        """
        Get random points on a Sphere
        Return Tensor of Size [x, 2, x ... x]
        """
        rand_grid = torch.cuda.FloatTensor(shape).to(device).float()
        rand_grid.data.uniform_(0, 1)
        return Variable(rand_grid)

    def get_regular_points(self, npoints=2500, device="gpu0"):
        """
        Get regular points on a Square
        Return Tensor of Size [x, 3]
        """
        if not self.npoints == npoints:
            self.npoints = npoints
            vertices, faces = self.generate_square(np.sqrt(npoints))
            self.mesh = pymesh.form_mesh(vertices=vertices, faces=faces)  # 10k vertices
            self.vertex = torch.from_numpy(self.mesh.vertices).to(device).float()
            self.num_vertex = self.vertex.size(0)
            self.vertex = self.vertex.transpose(0,1).contiguous().unsqueeze(0)

        return Variable(self.vertex[:, :2].contiguous().to(device))

    @staticmethod
    def generate_square(grain):
        """
        Generate a square mesh from a regular grid.
        :param grain:
        :return:
        """
        grain = int(grain)
        grain = grain - 1  # to return grain*grain points
        # generate regular grid
        faces = []
        vertices = []
        for i in range(0, int(grain + 1)):
            for j in range(0, int(grain + 1)):
                vertices.append([i / grain, j / grain, 0])

        for i in range(1, int(grain + 1)):
            for j in range(0, (int(grain + 1) - 1)):
                faces.append([j + (grain + 1) * i,
                              j + (grain + 1) * i + 1,
                              j + (grain + 1) * (i - 1)])
        for i in range(0, (int((grain + 1)) - 1)):
            for j in range(1, int((grain + 1))):
                faces.append([j + (grain + 1) * i,
                              j + (grain + 1) * i - 1,
                              j + (grain + 1) * (i + 1)])

        return np.array(vertices), np.array(faces)
