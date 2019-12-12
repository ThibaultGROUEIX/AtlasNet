# Written by the mighty Pierre-Alain Langlois


import argparse
from os import listdir
from os.path import isfile, join
import pymesh
import numpy as np
import copy


def shuffle_pc(file, output_path):
    """
    Function to shuffle a point cloud produced by virtual scanner.
    """
    mesh = pymesh.load_mesh(file)
    vertices = copy.deepcopy(mesh.vertices)
    permutation = np.random.permutation(len(vertices))
    vertices = vertices[permutation]
    new_mesh = pymesh.meshio.form_mesh(vertices, mesh.faces)
    new_mesh.add_attribute("vertex_nx")
    new_mesh.set_attribute("vertex_nx", mesh.get_vertex_attribute("vertex_nx")[permutation])
    new_mesh.add_attribute("vertex_ny")
    new_mesh.set_attribute("vertex_ny", mesh.get_vertex_attribute("vertex_ny")[permutation])
    new_mesh.add_attribute("vertex_nz")
    new_mesh.set_attribute("vertex_nz", mesh.get_vertex_attribute("vertex_nz")[permutation])
    pymesh.save_mesh(output_path, new_mesh, ascii=True, anonymous=True, use_float=True, *new_mesh.get_attribute_names())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input file", required=True)
    parser.add_argument("--output", help="Output file", required=True)
    args = parser.parse_args()
    shuffle_pc(args.input, args.output)


if __name__ == "__main__":
    main()
