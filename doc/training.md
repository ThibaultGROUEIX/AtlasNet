# Training

:raised_hand_with_fingers_splayed: Monitor your training on http://localhost:8888/

![visdom](./pictures/visdom2.png)



## Options

```shell
# Training parameters
--no_learning, action="store_true", help="Learning mode (batchnorms...)"
--batch_size, type=int, default=32, help="input batch size"
--batch_size_test, type=int, default=32, help="input batch size"
--workers, type=int, help="number of data loading workers', default=0)"
--nepoch, type=int, default=150, help="number of epochs to train for"
--start_epoch, type=int, default=0, help="number of epochs to train for"
--randomize, action="store_true", help="Fix random seed or not"
--lrate, type=float, default=0.001, help="learning rate"
--lr_decay_1, type=int, default=120, help="learning rate decay 1"
--lr_decay_2, type=int, default=140, help="learning rate decay 2"
--lr_decay_3, type=int, default=145, help="learning rate decay 2"
--run_single_eval, action="store_true", help="evaluate a trained network"
--demo, action="store_true", help="run demo autoencoder or single-view"
--multi_gpu, nargs='+', type=int, default=[0], help="use multiple gpus'"
--loop_per_epoch, type=int, default=1, help="number of data loop per epoch"



# Data
--shapenet13, action="store_true", help="Load 13 usual shapenet categories"
--SVR, action="store_true", help="Single_view Reconstruction"
--sample, action="store_false", help="Sample the input pointclouds"
--class_choice, nargs='+', default=["airplane"], type=str)
--number_points, type=int, default=2500, help="Number of point sampled on the object"
--random_rotation, action="store_true", help="apply data augmentation : random rotation"
--dataset, type=str, default="Shapenet",
                    choices=['Shapenet'])
--normalization, type=str, default="UnitBall",
                    choices=['UnitBall', 'BoundingBox', 'Identity'])
--number_points_eval, type=int, default=2500,
                    help="Number of point sampled on the object during evaluation"
--data_augmentation_axis_rotation, action="store_true",
                    help="apply data augmentation : axial rotation "
--data_augmentation_random_flips, action="store_true",
                    help="apply data augmentation : random flips"
--random_translation, action="store_true",
                    help="apply data augmentation :  random translation "
--anisotropic_scaling, action="store_true",
                    help="apply data augmentation : anisotropic scaling"

# Save dirs and reload
--id, type=str, default="0", help="training name"
--env, type=str, default="Atlasnet", help="visdom environment"
--port, type=int, default=8890, help="visdom port"
--dir_name, type=str, default="", help="dirname"
--demo_path, type=str, default="./doc/pictures/plane_input_demo.png", help="dirname"

# Network
--model, type=str, default='', help="optional reload model path"
--num_layers, type=int, default=2, help="number of hidden MLP Layer"
--hidden_neurons, type=int, default=512, help="number of neurons in each hidden layer"
--nb_primitives, type=int, default=1, help="number of primitives"
--remove_all_batchNorms, action="store_true", help="Replace all batchnorms by identity"
--bottleneck_size, type=int, default=1024, help="dim_out_patch"
--activation, type=str, default='relu',
                    choices=["relu", "sigmoid", "softplus", "logsigmoid", "softsign", 											"tanh"], help="dim_out_patch"
--template_type, type=str, default="SQUARE", choices=["SPHERE", "SQUARE"],
                    help="dim_out_patch"
# Loss
--compute_metro, action="store_true", help="Compute metro distance"

```





##Quantitative Results 


The number reported are the chamfer distance, the f-score and the [metro](https://github.com/ThibaultGROUEIX/AtlasNet/issues/34) distance.

| Method                 | Chamfer⁽⁰⁾ | Fscore | Metro | GPU memory | Total Train time |
| ---------------------- | ---------- | ------ | ----- | ---------- | ---------------- |
| Autoencoder 25 Squares | -          | -      | -     | -          | -                |
| Autoencoder 1 Sphere   | -          | -      | -     | -          | -                |
| SingleView 25  Squares | -          | -      | -     | -          | -                |
| SingleView 1 Sphere    | -          | -      | -     | -          | -                |

⁽⁰⁾  computed between 2500 ground truth points and 2500 reconstructed points.

⁽¹⁾ with the flag ```--accelerated_chamfer 1```.

⁽²⁾this is only an estimate, the code is not optimised.  The easiest way to enhance it would be to preload the training data to use the GPU at 100%. Time computed with the flag ```--accelerated_chamfer 1```.
Visualisation 





# Paper reproduction 

To reproduce main results from the paper : ```python ./training/launch.py --mode train```

In case you need the results of ICP on PointSetGen output :

* [ICP on PSG](https://cloud.enpc.fr/s/3a7Xg9RzIsgmofw)

