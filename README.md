# AtlasNet

This repository contains the source codes for the paper [A Papier-Mâché Approach to Learning Mesh Synthesis](http://imagine.enpc.fr/~groueixt/atlasnet/). The network is able to synthesize a mesh (point cloud + connectivity) from a low-resolution point cloud, or from an image.

​	TODO : add cool gifs

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

## Install

This implementation uses [Pytorch](http://pytorch.org/). Please note that the Chamfer Distance code doesn't work on [Pytorch v2](http://pytorch.org/) because of some weird error with the batch norm layers. It has been tested on v1.12, v3 and the latest sources available to date. TODO : update specfiles

```shell
## Download the repository
git clone git@github.com:ThibaultGROUEIX/PointSetGen-pytorch.git
conda create --name pytorch-atlasnet --file spec-file.txt
source activate pytorch-atlasnet
```

## Data and Trained models

We used the [ShapeNet](https://www.shapenet.org/) dataset for 3D models, and rendered views from [3D-R2N2](https://github.com/chrischoy/3D-R2N2):

* [The point clouds from ShapeNet, with normals](https://mega.nz/#!9LhW2CxT!A9d45cri4q8q10HfukUV_cy7J1lbWTFQtw7DKJlZKKAhttps://mega.nz/#!9LhW2CxT!A9d45cri4q8q10HfukUV_cy7J1lbWTFQtw7DKJlZKKA) go in ``` data/customShapeNet```
* [The corresponding normalized mesh (for the metro distance)](https://mega.nz/#!leAFEK5T!SDrcll-caO4p8ws7zDNKPpjNNWEMcf9AQ-rmR79t_OA) go in ``` data/ShapeNetCorev2Normalized```
* [the rendered views](https://mega.nz/#!4TgzCYTB!ACfHTD9VpUSUYYwI75k-GrSdqMH19jK0-CwBg1wKH08) go in ``` data/ShapeNetRendering```

The trained models and some corresponding results are also available online :

* trained_models (TODO)

## Demo

```shell
./scripts/demo_SVR.sh
```

 TODO : Here show some results



## Train

* First launch a visdom server :

```bash
python -m visdom.server -p 8888
```

* Launch the training

```shell
git addexport CUDA_VISIBLE_DEVICES=0 #whichever you want
source activate pytorch-atlasnet
git pull
env=AE_AtlasNet
nb_primitives=25
python ./training/train_AE_AtlasNet.py --env $env --nb_primitives $nb_primitives |& tee ${env}.txt
```

* Monitor your training on http://localhost:8888/

![visdom](pictures/visdom.png)


* Compute some results with your trained model

  ```bash
  python ./inference/run_AE_AtlasNet.py
  ```

## License

[MIT](https://github.com/ThibaultGROUEIX/AtlasNet/blob/master/license_MIT)
