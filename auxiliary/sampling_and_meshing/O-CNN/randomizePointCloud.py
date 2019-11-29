import argparse
from os import listdir
from os.path import isfile, join
import pymesh
import numpy as np
import copy
import joblib
from joblib import Parallel, delayed
from collections import defaultdict
from os.path import join, basename, relpath, dirname, exists
import os
from easydict import EasyDict
from pcSamplingInfRayShapeNet import Count


def shuffle_pc(file, output_path, limit=None, count=None):
    try:
        if not os.path.exists(output_path + ".npy"):
            mesh = pymesh.load_mesh(file)

            vertices = copy.deepcopy(mesh.vertices)
            permutation = np.random.permutation(len(vertices))
            if limit is not None:
                permutation = permutation[:min(limit, len(permutation))]
            vertices = vertices[permutation]
            np.save(output_path + ".npy", vertices)
    except:
        Count.add(file)
        print(Count.failed_example_path)


def shuffle_folder(args):
    ply_classes = "/".join([args.shapenet_path, "ply"])
    npy_classes = "/".join([args.shapenet_path, "npy"])

    limit = None if args.limit == 0 else args.limit
    if not exists(npy_classes):
        os.makedirs(npy_classes)

    classes = [x for x in next(os.walk(ply_classes))[1]]

    for obj_class in classes:
        print(obj_class)
        npy_path = join(npy_classes, obj_class)
        ply_path = join(ply_classes, obj_class)
        if not exists(npy_path):
            os.makedirs(npy_path)

        onlyfiles = [(join(ply_path, f), join(npy_path, f), limit) for f in listdir(ply_path) if
                     isfile(join(ply_path, f))]

        class BatchCompletionCallBack(object):
            completed = defaultdict(int)

            def __init__(self, time, index, parallel):
                self.index = index
                self.parallel = parallel

            def __call__(self, index):
                BatchCompletionCallBack.completed[self.parallel] += 1
                print("Progress : %s %% " %
                      str(BatchCompletionCallBack.completed[self.parallel] * 100 / len(onlyfiles)))
                if self.parallel._original_iterator is not None:
                    self.parallel.dispatch_next()

        joblib.parallel.BatchCompletionCallBack = BatchCompletionCallBack

        _ = Parallel(n_jobs=-1, backend="multiprocessing") \
            (delayed(shuffle_pc)(*i) for i in onlyfiles)

    print(f"{Count.failed_example} failed examples")
    print(Count.failed_example_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapenet_path", help="Input Folder", required=True)
    parser.add_argument("--limit", help="Max number of points in the output cloud", default=0, type=int)
    args = parser.parse_args()
    shuffle_folder(EasyDict(args.__dict__))


if __name__ == "__main__":
    main()
