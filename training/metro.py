import sys
import argparse
import numpy as np
import pymesh
from os.path import exists
import os
import subprocess


def metro(path1, path2, metro='./auxiliary/metro_sources/build/metro'):
    print(f"calculing {path1}")
    cmd = f"{metro} {path1} {path2}"
    returned_output = subprocess.check_output(cmd, shell=True)
    returned_output = returned_output.decode("utf-8")
    location = returned_output.find("Hausdorff")
    returned_output = returned_output[location:location + 40]
    print(f"calculing {path1} Done !")

    return float(returned_output.split(" ")[2])


def isolate_files():
    with open('./inference/files-metro.txt', 'r') as file:
        files = file.read().split('\n')
    for file in files:
        if file[-3:] == "ply":
            cat = file.split('/')[0]
            name = file.split('/')[1][:-4]
            path_points = '/'.join(['.', 'dataset', 'data', 'ShapeNetV1PointCloud', cat, name + '.points.ply.npy'])
            path_obj = '/'.join(['', 'home', 'thibault', 'hdd', 'data', 'ShapeNetCore.v1', cat, name, 'model.obj'])
            mesh = pymesh.load_mesh(path_obj)
            points = np.load((path_points))
            if not exists('/'.join(['.', 'dataset', 'data', 'metro_files', cat])):
                os.mkdir('/'.join(['.', 'dataset', 'data', 'metro_files', cat]))

            pymesh.save_mesh('/'.join(['.', 'dataset', 'data', 'metro_files', cat, name + '.ply']), mesh, ascii=True)
            np.save('/'.join(['.', 'dataset', 'data', 'metro_files', cat, name + '.npy']), points)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path1', help="Input file", required=True)
    parser.add_argument('--path2', help="Input file", required=True)
    parser.add_argument('--metro', type=str, help='Path to the metro executable',
                        default='./metro_sources/build/metro')

    args = parser.parse_args()
    return metro(args.path1, args.path2, args.metro)


if __name__ == '__main__':
    a = main()
    print(a)
