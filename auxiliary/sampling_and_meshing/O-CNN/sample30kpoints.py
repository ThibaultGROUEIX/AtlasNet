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


def test_pc(file):
    try:
        points = np.load(file)
        if points.shape[0] != 30000:
            print(f"sample {file}")
            choice = np.random.choice(points.shape[0], 30000, replace=True)
            points = points[choice, :]
            np.save(file, points)

    except:
        Count.add(file)
        print(Count.failed_example_path)


def test_folder(args):
    npy_classes = "/".join([args.shapenet_path, "npy"])

    if not exists(npy_classes):
        os.makedirs(npy_classes)

    classes = [x for x in next(os.walk(npy_classes))[1]]

    for obj_class in classes:
        print(obj_class)
        npy_path = join(npy_classes, obj_class)
        if not exists(npy_path):
            os.makedirs(npy_path)

        onlyfiles = [join(npy_path, f) for f in listdir(npy_path) if
                     isfile(join(npy_path, f))]

        class BatchCompletionCallBack(object):
            completed = defaultdict(int)

            def __init__(self, time, index, parallel):
                self.index = index
                self.parallel = parallel

            def __call__(self, index):
                BatchCompletionCallBack.completed[self.parallel] += 1
                if self.parallel._original_iterator is not None:
                    self.parallel.dispatch_next()

        joblib.parallel.BatchCompletionCallBack = BatchCompletionCallBack

        _ = Parallel(n_jobs=1, backend="multiprocessing") \
            (delayed(test_pc)(i) for i in onlyfiles)

    print(f"{Count.failed_example} failed examples")
    print(Count.failed_example_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapenet_path", help="Input Folder", required=True)
    args = parser.parse_args()
    test_folder(EasyDict(args.__dict__))


if __name__ == "__main__":
    main()
