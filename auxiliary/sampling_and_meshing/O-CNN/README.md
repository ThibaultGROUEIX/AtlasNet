# Fork of O-CNN: Octree-based Convolutional Neural Networks 

## [Virtual Scanner](https://github.com/wang-ps/O-CNN/tree/master/virtual_scanner)

`cd O-CNN/virtual_scanner/`.

This folder contains the code for converting 3D models to dense point clouds with normals. It outputs a mesh in the `PLY` format.

Install

```
apt-get install -y --no-install-recommends libboost-all-dev libcgal-dev libeigen3-dev
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

Usage

```
    VirtualScanner <file_name> [nviews] [flags] [normalize]
        file_name: the name of the file (*.obj; *.off) to be processed.
        nviews: the number of views for scanning. Default: 6
        flags: Indicate whether to output normal flipping flag. Default: 0
        normalize: Indicate whether to normalize input mesh. Default: 0
Example:
    VirtualScanner input.obj 30         // process the file input.obj
```

Easily convert ply to obj

```
apt-get install assimp-utils
assimp export test.ply test.obj
```



## Point shuffling

Shuffle points sampled from Virtual Scanner. Can be useful to quickly load a subsampled version on the pointcloud (just read a random block of lines in a PLY). 

Dependencies : `pip install argparse joblib` and install [Pymesh](https://github.com/PyMesh/PyMesh)

```

#Usage on a folder of files
python randomizePointCloud.py --shapenet_path path
```



## Both in one line

```bash
python process_raw_obj.py --shapenet_path path
```



