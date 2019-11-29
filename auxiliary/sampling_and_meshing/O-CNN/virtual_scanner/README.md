# Virtual scanner for converting 3D model to point cloud


This folder contains the code for converting the 3D models to dense point clouds with normals (\*.points). As detailed in our paper, we build a virtual scanner and shoot rays to calculate the intersection point and oriented normal. 

The code is based on [Boost](https://www.boost.org/), [CGAL](http://www.cgal.org/) and the [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) libraries. After configuring these three libraries properly, the code can be built with visual studio easily.

`Note`: 
1. Sometimes, the executive file might collapse when the scale of the mesh is very large. This is one bug of CGAL. In order to mitigate this you can run VirtualScanner with the normalize flag set to 1.
2. The format of some off files in the `ModelNet40` is invalid. Before running the virtual scanner, fix the `ModelNet40` with this [script](https://github.com/Microsoft/O-CNN/blob/master/ocnn/octree/python/ocnn/utils/off_utils.py).


## Running Virtual Scanner
### Executable
The pre-built executive file is contained in the folder `prebuilt_binaries`, which has been test on the Win10 x64 system.

    Usage:
        VirtualScanner.exe <file_name> [nviews] [flags] [normalize]
            file_name: the name of the file (*.obj; *.off) to be processed.
            nviews: the number of views for scanning. Default: 6
            flags: Indicate whether to output normal flipping flag. Default: 0
            normalize: Indicate whether to normalize input mesh. Default: 0
    Example:
        VirtualScanner.exe input.obj 14         // process the file input.obj
        VirtualScanner.exe D:\data\ 14          // process all the obj/off files under the folder D:\Data


### Python
You can also use python to convert 'off' and 'obj' files.

    Example usage (single file):
        # Converts obj/off file to points
        from ocnn.virtualscanner import VirtualScanner
        scanner = VirtualScanner(filepath="input.obj", view_num=14, flags=False, normalize=True)
        scanner.save(output_path="output.points")

    Example usage (directory tree):
        # Converts all obj/off files in directory tree to points
        from ocnn.virtualscanner import DirectoryTreeScanner
        scanner = DirectoryTreeScanner(view_num=6, flags=False, normalize=True)
        scanner.scan_tree(input_base_folder='/ModelNet10', output_base_folder='/ModelNet10Points', num_threads=8)


## Output of Virtual Scanner
The result is in the format of `points`, which can be parsed with the following:

### CPP
```cpp
#include "virtual_scanner/points.h"

// ...
// Specify the filename of the points
string filename = "your_pointcloud.points";

// Load points
Points points;
points.read_points(filename)

// Point number
int n =  points.info().pt_num();

// Whether does the file contain point coordinates?
bool has_points = points.info().has_property(PtsInfo::kPoint);
// Get the pointer to points: x_1, y_1, z_1, ..., x_n, y_n, z_n
const float* ptr_points = points.ptr(PtsInfo::kPoint);

// Whether does the file contain normals?
bool has_normals = points.info().has_property(PtsInfo::kNormal);
// Get the pointer to normals: nx_1, ny_1, nz_1, ..., nx_n, ny_n, nz_n
const float* ptr_points = points.ptr(PtsInfo::kNormal);

// Whether does the file contain per-point labels?
bool has_labels = points.info().has_property(PtsInfo::kLabel);
// Get the pointer to labels: label_1, label_2, ..., label_n
const float* ptr_labels = points.ptr(PtsInfo::kLabel);
```

### Python
The Microsoft repo [O-CNN](https://github.com/Microsoft/O-CNN) contains the `ocnn.ocnn_base` package which defines a `Points` class under `ocnn.octree`. You can use this class to manipulate the points files and generate octrees.

## Building/Installing
### Building On Windows
To build in Windows you can,

1. Edit the project files to point to Boost, Eigen and CGAL,

or 

2. Use [Vcpkg](https://github.com/Microsoft/vcpkg) to install/build all the dependencies (note this takes a long time).
  ```
  git clone https://github.com/Microsoft/vcpkg
  cd vcpkg
  .\bootstrap-vcpkg.bat
  .\vcpkg integrate install
  .\vcpkg install cgal eigen3 boost-system boost-filesystem --triplet x64-windows
  ```
  Then to build, you can use the supplied solution file VirtualScanner.sln


### Building On Ubuntu
To build with ubuntu, you can use apt for the dependencies.
```
apt-get install -y --no-install-recommends libboost-all-dev libcgal-dev libeigen3-dev
```
Then you can use cmake to build the executable
From this project's directory,
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```
### Installing Python Package
To install the python package you need to ensure that Eigen and CGAL can be found by cmake. If you used Vcpkg or apt-get to retrieve those libraries it should automatically find it.

With that ensured,

**Dependencies install via VCPKG**
```
pip install scikit-build cmake Cython
pip install --install-option="--" --install-option="-DCMAKE_TOOLCHAIN_FILE=<VCPKG_DIRECTORY>\scripts\buildsystems\vcpkg.cmake" .
```
Where <VCPKG_DIRECTORY> is the directory you install VCPKG.

**Dependencies install via apt-get**
```
pip install scikit-build cmake Cython
pip install .
```

If you use our code, please cite our paper.

    @article {Wang-2017-OCNN,
        title     = {O-CNN: Octree-based Convolutional Neural Networks for 3D Shape Analysis},
        author    = {Wang, Peng-Shuai and Liu, Yang and Guo, Yu-Xiao and Sun, Chun-Yu and Tong, Xin},
        journal   = {ACM Transactions on Graphics (SIGGRAPH)},
        volume    = {36},
        number    = {4},
        year      = {2017},
    }

