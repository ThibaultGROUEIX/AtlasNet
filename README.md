# PointSetGen-pytorch

This repository contains the source codes for the paper [A Papier-Mâché Approach to Learning Mesh Synthesis](). The network is able to synthesize a mesh (point cloud + connectivity) from a low-resolution point cloud, or from an image.

## Citing this work

If you find this work useful in your research, please consider citing:

```
#####EDIT
@inproceedings{choy20163d,
  title={3D-R2N2: A Unified Approach for Single and Multi-view 3D Object Reconstruction},
  author={Choy, Christopher B and Xu, Danfei and Gwak, JunYoung and Chen, Kevin and Savarese, Silvio},
  booktitle = {Proceedings of the European Conference on Computer Vision ({ECCV})},
  year={2016}
}

```

## Project Page

The project page is available at : TODO

##Install

This implementation uses [Pytorch v1.12](http://pytorch.org/). Please note that the Chamfer Distance code doesn't work on the latest version of [Pytorch](http://pytorch.org/) because of some retro compatibility issues.

If one wants to use the demo, it's not an issue, though, if one wants to train the networks, on has the install the right version of [Pytorch](http://pytorch.org/).

```shell
## Download the repository
git clone git@github.com:ThibaultGROUEIX/PointSetGen-pytorch.git
conda create --name pytorch-meshnet --file spec-file.txt
source activate pytorch-meshnet
```

## Data and Trained models

We used the [ShapeNet](https://www.shapenet.org/) dataset for 3D models, and rendered views from [3D-R2N2](https://github.com/chrischoy/3D-R2N2):

To download the data, just run :

```shell
./scripts/download_data.sh
```



## Demo

```shell
./scripts/demo_SVR.sh
```

 Here show some results



## Train

```shell
./scripts/train_SVR.sh
```



## License

TODO
