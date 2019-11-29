import pymesh
import numpy as np
import torch
from torch.autograd import Variable


def GetTemplate(template_type, device=0):
    if template_type == "SQUARE":
        return SquareTemplate(device=device)
    elif template_type == "SPHERE":
        return SphereTemplate(device=device)
    else:
        print("select valid template type")


class SphereTemplate(object):
    def __init__(self, device=0, grain=6):
        self.device = device
        self.dim = 3

    def get_random_points(self, shape,  device="gpu0"):
        rand_grid = torch.cuda.FloatTensor(shape).to(device).float()
        rand_grid.data.normal_(0, 1)
        rand_grid = rand_grid / torch.sqrt(torch.sum(rand_grid ** 2, dim=1, keepdim=True))
        return Variable(rand_grid)

    def get_regular_points(self, npoints=None,  device="gpu0"):
        self.mesh = pymesh.generate_icosphere(1, [0, 0, 0], 4)  # 2562 vertices
        self.vertex = torch.from_numpy(self.mesh.vertices).to(device).float()
        self.num_vertex = self.vertex.size(0)
        return Variable(self.vertex)


class SquareTemplate(object):
    def __init__(self, device=0):
        self.device = device
        self.dim = 2

    def get_random_points(self, shape, device="gpu0"):
        rand_grid = torch.cuda.FloatTensor(shape).to(device).float()
        rand_grid.data.uniform_(0, 1)
        return Variable(rand_grid)

    def get_regular_points(self, npoints=2500, device="gpu0"):
        vertices, faces = self.generate_square(np.sqrt(npoints))
        self.mesh = pymesh.form_mesh(vertices=vertices, faces=faces)  # 10k vertices
        self.vertex = torch.from_numpy(self.mesh.vertices).to(device).float()
        self.num_vertex = self.vertex.size(0)
        return Variable(self.vertex[:, :2].contiguous())

    @staticmethod
    def generate_square(grain):
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
