# Demo

```python train.py --demo --demo_input_path YOUR_IMAGE_or_OBJ_PATH --reload_model_path YOUR_MODEL_PTH_PATH ```

```
This function takes an image or pointcloud path as input and save the mesh infered by Atlasnet
Extension supported are `ply` `npy` `obg` and `png`
--demo_input_path input file e.g. image.png or object.ply 
--reload_model_path trained model path (see below for pretrained models) 
:return: path to the generated mesh
```



To generate the example below, use `python train.py --demo`. It will default to the 2D plane image as input, download a trained single-view Altasnet with 25 square primitives, run the image through the network and save the generated 3D plane in `doc/pictures/`.



![input](./pictures/2D3D.png)

### Trained models

All training options can be recovered in `{dir_name}/options.txt`.
Trained are automatically downloaded by `train.py --demo`. Use `chmod +x training/download_trained_models.sh; ./training/download_trained_models.sh` to explicitly get the trained models.

* `./training/trained_models/atlasnet_autoencoder_25_squares/network.pth` [Default]

* `./training/trained_models/atlasnet_autoencoder_1_sphere/network.pth` 

* `./training/trained_models/atlasnet_singleview_25_squares/network.pth` [Default]

* `./training/trained_models/atlasnet_singleview_1_sphere/network.pth` 

  




You can use our  [Meshlab Visualization Trick](./doc/meshlab.md) to have nicer visualization of the generated mesh in Meshlab.
