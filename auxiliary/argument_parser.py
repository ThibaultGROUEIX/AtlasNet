import argparse
import auxiliary.my_utils as my_utils
import os
import datetime
import json
from termcolor import colored
from easydict import EasyDict


def Args2String(opt):
    my_str = ""
    for i in opt.__dict__.keys():
        if i == "model":
            if opt.__dict__[i] is None:
                my_str = my_str + str(0) + "_"
            else:
                my_str = my_str + str(1) + "_"
        else:
            my_str = my_str + str(opt.__dict__[i]) + "_"
    my_str = my_str.replace('/', '-')
    return my_str


def parser():
    parser = argparse.ArgumentParser()
    # Training parameters
    parser.add_argument('--learning', type=int, default=1, help='input batch size')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--batch_size_test', type=int, default=32, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--start_epoch', type=int, default=0, help='number of epochs to train for')
    parser.add_argument('--randomize', type=int, default=0,
                        help='if 1, projects predicted correspondences point on target mesh')
    parser.add_argument('--lrate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay_1', type=int, default=80, help='learning rate decay 1')
    parser.add_argument('--lr_decay_2', type=int, default=90, help='learning rate decay 2')
    parser.add_argument('--lr_decay_3', type=int, default=95, help='learning rate decay 2')
    parser.add_argument('--run_single_eval', type=int, default=0, help='learning rate decay 2')

    # Data
    parser.add_argument('--dataset', type=str, default="Shapenet",
                        choices=['Shapenet'])
    parser.add_argument('--normalization', type=str, default="UnitBall",
                        choices=['UnitBall', 'BoundingBox', 'Identity'])
    parser.add_argument('--shapenet13', type=int, default=1)
    parser.add_argument('--SVR', type=int, default=0)
    parser.add_argument('--sample', type=int, default=1)
    parser.add_argument('--class_choice', nargs='+', default=["airplane"], type=str)
    parser.add_argument('--number_points', type=int, default=2500, help='Number of point sampled on the object')
    parser.add_argument('--number_points_eval', type=int, default=2500, help='Number of point sampled on the object during evaluation')
    parser.add_argument('--random_rotation', type=int, default=0, help='Number of point sampled on the object')
    parser.add_argument('--data_augmentation_axis_rotation', type=int, default=0, help='Faust eval')
    parser.add_argument('--data_augmentation_random_flips', type=int, default=0, help='Faust eval')
    parser.add_argument('--random_translation', type=int, default=0, help='Faust eval')
    parser.add_argument('--anisotropic_scaling', type=int, default=0, help='Faust eval')

    # Save dirs and reload
    parser.add_argument('--id', type=str, default="0", help='training name')
    parser.add_argument('--env', type=str, default="Atlasnet", help='visdom environment')
    parser.add_argument('--display', type=int, default=1, help='visdom environment')
    parser.add_argument('--port', type=int, default=8890, help='visdom port')
    parser.add_argument('--dir_name', type=str, default="", help='dirname')

    # Network
    parser.add_argument('--model', type=str, default='', help='optional reload model path')
    parser.add_argument('--loop_per_epoch', type=int, default=1, help='optional reload model path')
    parser.add_argument('--nb_primitives', type=int, default=1, help='number of primitives')
    parser.add_argument('--template_type', type=str, default="SQUARE", choices=["SPHERE", "SQUARE"],
                        help='dim_out_patch')
    parser.add_argument('--decoder_type', type=str, default="AtlasNet", choices=["AtlasNet", "AtlasNetLight"],
                        help='dim_out_patch')
    parser.add_argument('--multi_gpu', nargs='+', type=int, default=[0], help='dim_out_patch')
    parser.add_argument('--remove_all_batchNorms', type=int, default=0, help='dim_out_patch')
    parser.add_argument('--bottleneck_size', type=int, default=1024, help='dim_out_patch')

    # Loss
    parser.add_argument('--compute_metro', type=int, default=1, help='dim_out_patch')

    opt = parser.parse_args()

    opt.randomize = my_utils.int_2_boolean(opt.randomize)
    opt.display = my_utils.int_2_boolean(opt.display)
    opt.random_rotation = my_utils.int_2_boolean(opt.random_rotation)
    opt.run_single_eval = my_utils.int_2_boolean(opt.run_single_eval)
    opt.remove_all_batchNorms = my_utils.int_2_boolean(opt.remove_all_batchNorms)
    opt.data_augmentation_axis_rotation = my_utils.int_2_boolean(opt.data_augmentation_axis_rotation)
    opt.data_augmentation_random_flips = my_utils.int_2_boolean(opt.data_augmentation_random_flips)
    opt.anisotropic_scaling = my_utils.int_2_boolean(opt.anisotropic_scaling)
    opt.random_translation = my_utils.int_2_boolean(opt.random_translation)
    opt.learning = my_utils.int_2_boolean(opt.learning)
    opt.shapenet13 = my_utils.int_2_boolean(opt.shapenet13)
    opt.SVR = my_utils.int_2_boolean(opt.SVR)
    opt.compute_metro = my_utils.int_2_boolean(opt.compute_metro)

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
    return opt
