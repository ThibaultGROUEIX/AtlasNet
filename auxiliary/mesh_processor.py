import pymesh
import numpy as np
from os.path import join, dirname


class ColorMap:
    def __init__(self):
        self.colormap_path = join(dirname(__file__), "colormap.npy")
        self.colormap = (np.load(self.colormap_path) * 255).astype('int')

    def __call__(self, index):
        """
        :param value: a float
        :return:
        """
        colors = self.colormap[index]
        return colors


def save(mesh, path, colormap):
    pymesh.save_mesh(path, mesh, ascii=True)
    try:
        vertex_sources = mesh.get_attribute("vertex_sources")  # batch, nb_prim, num_point, 3
        vertex_sources = (255 * (vertex_sources / vertex_sources.max())).astype('int')
        mesh.add_attribute("vertex_red")
        mesh.add_attribute("vertex_green")
        mesh.add_attribute("vertex_blue")
        mesh.set_attribute("vertex_red", colormap.colormap[vertex_sources][:, 0])
        mesh.set_attribute("vertex_green", colormap.colormap[vertex_sources][:, 1])
        mesh.set_attribute("vertex_blue", colormap.colormap[vertex_sources][:, 2])
        pymesh.save_mesh(path[:-3] + "ply", mesh, *mesh.get_attribute_names(), ascii=True)
    except:
        print("could not save mesh with colors")
