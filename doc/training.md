# Training

##### Autoencoder [default]
```shell
python train.py --shapenet13 --dir_name log/atlasnet_autoencoder_25_squares --nb_primitives 25 --template_type "SQUARE" 
```



##### Single-View Reconstruction [default]

```shell
python train.py --shapenet13 --dir_name log/atlasnet_singleview_25_squares_tmp --nb_primitives 25 --template_type SQUARE --SVR --reload_decoder_path log/atlasnet_autoencoder_25_squares --train_only_encoder 
```



:raised_hand_with_fingers_splayed: Monitor your training on http://localhost:8890/

:raised_hand_with_fingers_splayed: See report on a completed training at http://localhost:8891/{DIR_NAME}



![visdom](./pictures/netvision.png)



## Options : --flag, default, *help*

##### Training parameters


* `--batch_size`, **32**,  *Training batch size*
* `--batch_size_test`, **32**,  *Testing batch size*
* `--workers`, **0**,  *Number of data loading workers*
* `--nepoch`, **150**, *Number of training epochs*
* `--start_epoch`, **0**, *Start directly from epoch [X]*
* `--random_seed`, **False**, *Fix random seed*
* `--lrate`, **0.001**, *Learning rate*
* `--lr_decay_1`, **120**, *Learning rate decay 1*
* `--lr_decay_2`, **140**, *Learning rate decay 2*
* `--lr_decay_3`, **145**, *Learning rate decay 3*
* `--multi_gpu`, **[0]**, *Replacing CUDA_VISIBLE_DEVICES=X. i.e  to use gpu 1 and 2 `--multi_gpu 0 1`*
* `--loop_per_epoch`, **1**, *Number of data loop per epoch*
* `--no_learning`,  **False**, *No backdrop + network in eval mode*
* `--train_only_encoder`, **False**, *Only train the encoder, freeze decoder*
* `--run_single_eval`, **False**, *Flag to evaluate a trained network reloaded with `--reload_model_path`*
* `--demo`, **False**, *Run forward pass on an input and save mesh output. Input provided by `--demo_input_path`, network reloaded with `--reload_model_path`*




##### Data
* `--SVR`, **False**, *Single_view Reconstruction mode*. *Affects dataloader and network architecture*
* `--shapenet13`, **False**, *Load the 13 usual shapenet categories*
* `--class_choice`, **["airplane"]**, *Choose classes to train on*
* `--random_rotation`, **False**, *Random rotation*
* `--normalization`, **"UnitBall"** ,  Choices are 'UnitBall', 'BoundingBox', or 'Identity'
* `--sample`, **True**, *Sample the input pointclouds*
* `--number_points`, **2500**, *Number of point sampled on the object*
* `--number_points_eval`, **2500**, *Number of points sampled on the object during evaluation*

* `--data_augmentation_axis_rotation`, **False**, *Axial rotation*
* `--data_augmentation_random_flips`, **False**, *Random flips*
* `--random_translation`, **False**, *Random translation*
* `--anisotropic_scaling`, **False**, *Anisotropic scaling*



##### Paths

* `--demo_input_path`, **"./doc/pictures/plane_input_demo.png"**, *Input test path, requires `--demo`*
* `--reload_model_path`, **"./training/trained_models/atlasnet_AE_25_patches.pth"**, *Path to pre-trained model*
* `--reload_decoder_path`, **"./training/trained_models/atlasnet_AE_25_patches.pth"**, *Path to pre-trained model*
* `--env`, **"Atlasnet"**, *Visdom environment*
* `--visdom_port`, **8890**, *Visdom port*
* `--http_port`, **8891**, *Http port*
* `--dir_name`, **""**, *Name of the log folder.*
* `--id`, **0**, *Training id*



##### Network

* `--nb_primitives`, **1**, *Number of primitives in atlasnet decoder*

* `--template_type`, **"SQUARE"**, *Choices are "SPHERE", or "SQUARE"*

* `--num_layers`, **2**, *Number of hidden MLP Layer in atlasnet decoder*

* `--hidden_neurons`, **512**, *Number of neurons in each hidden layer of atlasnet decoder*

* `--bottleneck_size`, **1024**, *Bottleneck_size for the auto encoder*

* `--activation`, **'relu'**, *Choices are "relu", "sigmoid", "softplus", "logsigmoid", "softsign", or "tanh"*

* `--remove_all_batchNorms`, **False**, *Replace all batchnorms by identity*

     

##### Loss

* `--no_metro`, **False**, *Skip metro distance* *evaluation*.




## Quantitative Results 


| Method                 | Chamfer (*1) | Fscore (*2) | [Metro](https://github.com/ThibaultGROUEIX/AtlasNet/issues/34) (*3) | Total Train time (min) |
| ---------------------- | ---- | ----   | ----- |-------     |
| Autoencoder 25 Squares | 1.35 | 82.3%   | 6.82  | 731       |
| Autoencoder 1 Sphere   | 1.35 | 83.3%   | 6.94  | 548    |
| SingleView 25  Squares | 3.78 | 63.1% | 8.94 | 1422      |
| SingleView 1 Sphere    | 3.76 | 64.4% |  9.01  | 1297      |


  * (*1) x1000. Computed between 2500 ground truth points and 2500 reconstructed points. 
  * (*2) The threshold is 0.001
  * (*3) x100. Metro is ran on unormalized point clouds (which explains a difference with the paper's numbers) 


# Paper reproduction 

To reproduce main results from the paper : ```python ./training/launcher.py```

In case you need the results of ICP on PointSetGen output :

* [ICP on PSG](https://cloud.enpc.fr/s/3a7Xg9RzIsgmofw)

