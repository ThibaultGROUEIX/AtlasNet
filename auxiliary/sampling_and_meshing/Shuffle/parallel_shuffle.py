# Written by the mighty Pierre-Alain Langlois


import argparse
from os import listdir
from os.path import isfile, join
import pymesh
import numpy as np
import copy
import joblib
from joblib import Parallel, delayed
from collections import defaultdict


def shuffle_pc(file, output_path):
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
    parser.add_argument("--input", help="Input Folder", required=True)
    parser.add_argument("--output", help="Output Folder", required=True)
    args = parser.parse_args()

    onlyfiles = [(join(args.input, f), join(args.output, f)) for f in listdir(args.input) if
                 isfile(join(args.input, f))]

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


if __name__ == "__main__":
    main()
