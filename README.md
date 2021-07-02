## AtlasNet [[Project Page]](http://imagine.enpc.fr/~groueixt/atlasnet/) [[Paper]](https://arxiv.org/abs/1802.05384) [[Talk]](http://imagine.enpc.fr/~groueixt/atlasnet/atlasnet_slides_spotlight_CVPR.pptx)

**AtlasNet: A Papier-Mâché Approach to Learning 3D Surface Generation** <br>
Thibault Groueix,  Matthew Fisher, Vladimir G. Kim , Bryan C. Russell, Mathieu Aubry  <br>
In [CVPR, 2018](http://cvpr2018.thecvf.com/).

:rocket: New branch : [AtlasNet + Shape Reconstruction by Learning Differentiable Surface Representations](https://github.com/ThibaultGROUEIX/AtlasNet/tree/jacobian_regularization)

<img src="doc/pictures/chair.png" alt="chair.png" width="35%" /> <img src="doc/pictures/chair.gif" alt="chair.gif" width="32%" />





### Install

This implementation uses Python 3.6, [Pytorch](http://pytorch.org/), [Pymesh](https://github.com/PyMesh/PyMesh), Cuda 10.1. 
```shell
# Copy/Paste the snippet in a terminal
git clone --recurse-submodules https://github.com/ThibaultGROUEIX/AtlasNet.git
cd AtlasNet 

#Dependencies
conda create -n atlasnet python=3.6 --yes
conda activate atlasnet
conda install  pytorch torchvision cudatoolkit=10.1 -c pytorch --yes
pip install --user --requirement  requirements.txt # pip dependencies
```



##### Optional : Compile Chamfer (MIT) + Metro Distance (GPL3 Licence)
```shell
# Copy/Paste the snippet in a terminal
python auxiliary/ChamferDistancePytorch/chamfer3D/setup.py install #MIT
cd auxiliary
git clone https://github.com/ThibaultGROUEIX/metro_sources.git
cd metro_sources; python setup.py --build # build metro distance #GPL3
cd ../..
```

### A note on data.

Data download should be automatic. However, due to the new google drive traffic caps, you may have to download manually. If you run into an error running the demo,
you can refer to #61. 

You can manually download the data from three sources (there are the same) :
* Google drive : https://drive.google.com/drive/folders/1If_-t0Aw9Zps-gj5ttgaMSTqRwYms9Ag?usp=sharing
* Kaggle : https://www.kaggle.com/thibeix/atlasnet-data
* NextCloud : https://cloud.enpc.fr/s/z9TxRcxGgeYGDJ4

Please make sure to unzip the archives in the right places :

```shell
cd AtlasNet
mkdir data
unzip ShapeNetV1PointCloud.zip -d ./data/
unzip ShapeNetV1Renderings.zip -d ./data/
unzip metro_files.zip -d ./data/
unzip trained_models.zip -d ./training/
```
### Usage


* **[Demo](./doc/demo.md)** :    ```python train.py --demo```
* **[Training](./doc/training.md)** :  ```python train.py --shapenet13```  *Monitor on  http://localhost:8890/*
* <details><summary> Latest Refacto 12-2019  </summary>
  - [x] Factorize Single View Reconstruction and autoencoder in same class <br>
  - [x] Factorise Square and Sphere template in same class<br>
  - [x] Add latent vector as bias after first layer(30% speedup) <br>
  - [x] Remove last th in decoder <br>
  - [x] Make large .pth tensor with all pointclouds in cache(drop the nasty Chunk_reader) <br>
  - [x] Make-it multi-gpu <br>
  - [x] Add netvision visualization of the results <br>
  - [x] Rewrite main script object-oriented  <br>
  - [x] Check that everything works in latest pytorch version <br>
  - [x] Add more layer by default and flag for the number of layers and hidden neurons <br>
  - [x] Add a flag to generate a mesh directly <br>
  - [x] Add a python setup install <br>
  - [x] Make sure GPU are used at 100% <br>
  - [x] Add f-score in Chamfer + report f-score <br>
  - [x] Get rid of shapenet_v2 data and use v1! <br>
  - [x] Fix path issues no more sys.path.append <br>
  - [x] Preprocess shapenet 55 and add it in dataloader <br>
  - [x] Make minimal dependencies <br>
  </details>

  

### Quantitative Results 


| Method                 | Chamfer (*1) | Fscore (*2) | [Metro](https://github.com/ThibaultGROUEIX/AtlasNet/issues/34) (*3) | Total Train time (min) |
| ---------------------- | ---- | ----   | ----- |-------     |
| Autoencoder 25 Squares | 1.35 | 82.3%   | 6.82  | 731       |
| Autoencoder 1 Sphere   | 1.35 | 83.3%   | 6.94  | 548    |
| SingleView 25  Squares | 3.78 | 63.1% | 8.94 | 1422      |
| SingleView 1 Sphere    | 3.76 | 64.4% |  9.01  | 1297      |


  * (*1) x1000. Computed between 2500 ground truth points and 2500 reconstructed points. 
  * (*2) The threshold is 0.001
  * (*3) x100. Metro is ran on unormalized point clouds (which explains a difference with the paper's numbers) 


### Related projects

*  [Learning Elementary Structures](https://github.com/TheoDEPRELLE/AtlasNetV2)
*  [3D-CODED](https://github.com/ThibaultGROUEIX/3D-CODED)
*  [Cycle Consistent Deformations](https://github.com/ThibaultGROUEIX/CycleConsistentDeformation)
*  [Atlasnet code V2.2](https://github.com/ThibaultGROUEIX/AtlasNet/tree/V2.2) (more linear, script like, may be easier to understand at first)





### Citing this work

```
@inproceedings{groueix2018,
          title={{AtlasNet: A Papier-M\^ach\'e Approach to Learning 3D Surface Generation}},
          author={Groueix, Thibault and Fisher, Matthew and Kim, Vladimir G. and Russell, Bryan and Aubry, Mathieu},
          booktitle={Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
          year={2018}
        }
```
<p align="center">
  <img  src="doc/pictures/plane.gif">
</p>
