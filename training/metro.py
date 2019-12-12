import argparse
import numpy as np
import pymesh
from os.path import exists
import os
import subprocess
from shutil import copy

"""
        Author : Thibault Groueix 01.11.2019
"""


def metro(path1, path2, metro='./auxiliary/metro_sources/build/metro'):
    """
    Run the metro compiled program on two meshes and get the output.
    :param path1: mesh 1
    :param path2: mesh 2
    :param metro: path to metro
    :return: metro(mesh 1, mesh 2) [float]
    """

    print(f"calculing {path1}")
    cmd = f"{metro} {path1} {path2}"
    returned_output = subprocess.check_output(cmd, shell=True)
    returned_output = returned_output.decode("utf-8")
    location = returned_output.find("Hausdorff")
    returned_output = returned_output[location:location + 40]
    distance = float(returned_output.split(" ")[2])
    print(f"calculing {path1} Done {distance}!")

    return distance


def isolate_files():
    """
    Utility fonction to generate the metro_file archive. Useless to all users but the author.
    """
    with open('./dataset/data/metro_files/files-metro.txt', 'r') as file:
        files = file.read().split('\n')
    for file in files:
        if file[-3:] == "ply":
            cat = file.split('/')[0]
            name = file.split('/')[1][:-4]
            path_points = '/'.join(['.', 'dataset', 'data', 'ShapeNetV1PointCloud', cat, name + '.points.ply.npy'])
            path_png = '/'.join(['.', 'dataset', 'data', 'ShapeNetV1Renderings', cat, name, "rendering", '00.png'])

            path_obj = '/'.join(['', 'home', 'thibault', 'hdd', 'data', 'ShapeNetCore.v1', cat, name, 'model.obj'])
            mesh = pymesh.load_mesh(path_obj)
            points = np.load((path_points))
            if not exists('/'.join(['.', 'dataset', 'data', 'metro_files', cat])):
                os.mkdir('/'.join(['.', 'dataset', 'data', 'metro_files', cat]))

            pymesh.save_mesh('/'.join(['.', 'dataset', 'data', 'metro_files', cat, name + '.ply']), mesh, ascii=True)
            np.save('/'.join(['.', 'dataset', 'data', 'metro_files', cat, name + '.npy']), points)
            copy(path_png, '/'.join(['.', 'dataset', 'data', 'metro_files', cat, name + '.png']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path1', help="Input file", required=True)
    parser.add_argument('--path2', help="Input file", required=True)
    parser.add_argument('--metro', type=str, help='Path to the metro executable',
                        default='./metro_sources/build/metro')

    args = parser.parse_args()
    return metro(args.path1, args.path2, args.metro)


if __name__ == '__main__':
    a = isolate_files()
    print(a)
