### AtlasNet + Shape Reconstruction by Learning Differentiable Surface Representations

Welcome to this branch : it is an unofficial implemention of [Shape Reconstruction by Learning Differentiable Surface Representations](https://arxiv.org/abs/1911.11227) from Yan Bednarik and Thibault Groueix.



### Note

This repo include an implementation from Yan Bednarik of Differentiable Surface properties and their usage to form a conformal regularization of 3D shape reconstruction (on ShapeNet).

Such properties include (please check Yan's paper for details):

* Normals 
* First fondamental form (Jacobian)
* Mean curvature
* Gaussian curvature

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



### Training

Same options as the main branch. See here [Training](./doc/training.md)   Just add `--conformal_regul --lambda_conformal_regul 0.001` to train with Differentiable Surface properties. 

##### Autoencoder [default]


```shell
python train.py --shapenet13 --dir_name log/atlasnet_autoencoder_25_squares --nb_primitives 25 --template_type "SQUARE" --conformal_regul --lambda_conformal_regul 0.001
```

##### Single-View Reconstruction [default]

```shell
python train.py --shapenet13 --dir_name log/atlasnet_singleview_25_squares_tmp --nb_primitives 25 --template_type SQUARE --SVR --reload_decoder_path log/atlasnet_autoencoder_25_squares --train_only_encoder --conformal_regul --lambda_conformal_regul 0.001
```

:raised_hand_with_fingers_splayed: Monitor your training on http://localhost:8890/

:raised_hand_with_fingers_splayed: See report on a completed training at http://localhost:8891/{DIR_NAME}



### Using the trained models

```python train.py --demo --demo_input_path YOUR_IMAGE_or_OBJ_PATH --reload_model_path YOUR_MODEL_PTH_PATH ```

```
This function takes an image or pointcloud path as input and save the mesh infered by Atlasnet
Extension supported are `ply` `npy` `obg` and `png`
--demo_input_path input file e.g. image.png or object.ply 
--reload_model_path trained model path (see below for pretrained models) 
:return: path to the generated mesh
```



### Quantitative Results [Pretrained models](https://drive.google.com/a/polytechnique.org/uc?id=1mlA57o7n7CK9u8RpYS_RekTQN7RTNjhl&export=download)


| Method (on Planes) | Chamfer (*1) | Fscore (*2) | [Metro](https://github.com/ThibaultGROUEIX/AtlasNet/issues/34) (*3) | Total Train time (min) |
| ---------------------- | ---- | ----   | ----- |-------     |
| Autoencoder 1 Square | **1.40** | **85.3** | **19** | **30** |
| Autoencoder 1 Square + Conformal (lamda=1) | 2.8 | 72.9 | 24 | 300 |
| Autoencoder 1 Square + Conformal (lamda=0.001) | 1.45 | 85.3 | 20 | 300 |

  * (*1) x1000. Computed between 2500 ground truth points and 2500 reconstructed points. 
  * (*2) The threshold is sqrt(0.001)
  * (*3) x100. Metro is ran on unormalized point clouds (which explains a difference with the paper's numbers) 



### Citing this work

```
@article{bednarik2019shape,
   title={Shape Reconstruction by Learning Differentiable Surface Representations},
 	 author={Bednarik, Jan and Parashar, Shaifali and Gundogdu, Erhan and Salzmann, Mathieu and Fua, Pascal},
   booktitle={Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
 	 year={2019}
}
```

```
@inproceedings{groueix2018,
          title={{AtlasNet: A Papier-M\^ach\'e Approach to Learning 3D Surface Generation}},
          author={Groueix, Thibault and Fisher, Matthew and Kim, Vladimir G. and Russell, Bryan and Aubry, Mathieu},
          booktitle={Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
          year={2018}
        }
```
