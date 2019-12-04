import argparse
import auxiliary.my_utils as my_utils
import os
import datetime
import json
from termcolor import colored
from easydict import EasyDict


def parser():
    parser = argparse.ArgumentParser()

    # Training parameters
    parser.add_argument("--no_learning", action="store_true", help="Learning mode (batchnorms...)")
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--batch_size_test', type=int, default=32, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--nepoch', type=int, default=150, help='number of epochs to train for')
    parser.add_argument('--start_epoch', type=int, default=0, help='number of epochs to train for')
    parser.add_argument("--randomize", action="store_true", help="Fix random seed or not")
    parser.add_argument('--lrate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay_1', type=int, default=120, help='learning rate decay 1')
    parser.add_argument('--lr_decay_2', type=int, default=140, help='learning rate decay 2')
    parser.add_argument('--lr_decay_3', type=int, default=145, help='learning rate decay 2')
    parser.add_argument("--run_single_eval", action="store_true", help="evaluate a trained network")
    parser.add_argument("--demo", action="store_true", help="run demo autoencoder or single-view")

    # Data
    parser.add_argument('--dataset', type=str, default="Shapenet",
                        choices=['Shapenet'])
    parser.add_argument('--normalization', type=str, default="UnitBall",
                        choices=['UnitBall', 'BoundingBox', 'Identity'])
    parser.add_argument("--shapenet13", action="store_true", help="Load 13 usual shapenet categories")
    parser.add_argument("--SVR", action="store_true", help="Single_view Reconstruction")
    parser.add_argument("--sample", action="store_false", help="Sample the input pointclouds")
    parser.add_argument('--class_choice', nargs='+', default=["airplane"], type=str)
    parser.add_argument('--number_points', type=int, default=2500, help='Number of point sampled on the object')
    parser.add_argument('--number_points_eval', type=int, default=2500,
                        help='Number of point sampled on the object during evaluation')
    parser.add_argument("--random_rotation", action="store_true", help="apply data augmentation : random rotation")
    parser.add_argument("--data_augmentation_axis_rotation", action="store_true",
                        help="apply data augmentation : axial rotation ")
    parser.add_argument("--data_augmentation_random_flips", action="store_true",
                        help="apply data augmentation : random flips")
    parser.add_argument("--random_translation", action="store_true",
                        help="apply data augmentation :  random translation ")
    parser.add_argument("--anisotropic_scaling", action="store_true",
                        help="apply data augmentation : anisotropic scaling")

    # Save dirs and reload
    parser.add_argument('--id', type=str, default="0", help='training name')
    parser.add_argument('--env', type=str, default="Atlasnet", help='visdom environment')
    parser.add_argument('--port', type=int, default=8890, help='visdom port')
    parser.add_argument('--dir_name', type=str, default="", help='dirname')
    parser.add_argument('--demo_path', type=str, default="./doc/pictures/plane_input_demo.png", help='dirname')

    # Network
    parser.add_argument('--model', type=str, default='', help='optional reload model path')
    parser.add_argument('--num_layers', type=int, default=2, help='number of hidden MLP Layer')
    parser.add_argument('--hidden_neurons', type=int, default=512, help='number of neurons in each hidden layer')
    parser.add_argument('--loop_per_epoch', type=int, default=1, help='number of data loop per epoch')
    parser.add_argument('--nb_primitives', type=int, default=1, help='number of primitives')
    parser.add_argument('--template_type', type=str, default="SQUARE", choices=["SPHERE", "SQUARE"],
                        help='dim_out_patch')
    parser.add_argument('--multi_gpu', nargs='+', type=int, default=[0], help='Use multiple gpus')
    parser.add_argument("--remove_all_batchNorms", action="store_true", help="Replace all batchnorms by identity")
    parser.add_argument('--bottleneck_size', type=int, default=1024, help='dim_out_patch')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=["relu", "sigmoid", "softplus", "logsigmoid", "softsign", "tanh"], help='dim_out_patch')

    # Loss
    parser.add_argument("--compute_metro", action="store_true", help="Compute metro distance")

    opt = parser.parse_args()

    opt.date = str(datetime.datetime.now())
    now = datetime.datetime.now()
    opt = EasyDict(opt.__dict__)

    if opt.dir_name == "":
        opt.dir_name = os.path.join('log', opt.id + now.isoformat())
    else:
        print("Modifying input arguments to match network in dirname")
        try:
            with open(os.path.join(opt.dir_name, "options.json"), 'r') as f:
                my_opt_dict = json.load(f)
            my_opt_dict.pop("run_single_eval")
            my_opt_dict.pop("learning")
            my_opt_dict.pop("demo")
            my_opt_dict.pop("demo_path")
            for key in my_opt_dict.keys():
                opt[key] = my_opt_dict[key]
            my_utils.cyan_print("PARAMETER: ")
            for a in my_opt_dict:
                print(
                    "         "
                    + colored(a, "yellow")
                    + " : "
                    + colored(str(my_opt_dict[a]), "cyan")
                )
        except:
            print("failed to reload parameters from option.txt, must be a new experiment")

    if opt.template_type == "SQUARE":
        opt.dim_template = 2
    if opt.template_type == "SPHERE":
        opt.dim_template = 3
    opt.env = opt.env + opt.dir_name.split('/')[-1]

    if opt.demo:
        ext = opt.demo_path.split('.')[-1]
        if ext == "ply" or ext == "npy" or ext == "obj":
            opt.SVR = False
        elif ext == "png":
            opt.SVR = True

    return opt
