import argparse
import os
import gpustat
import time


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="current", choices=['training', 'inference', 'current'])
    opt = parser.parse_args()
    return opt


opt = parser()


class Experiments(object):
    def __init__(self):
        self.template = {
            1: "python train.py --dir_name log/template_sphere --template_type SPHERE",
            2: "python train.py --dir_name log/template_square --nb_primitives 1",
        }
        self.num_prim = {
            1: "python train.py --dir_name log/num_prim_10 --nb_primitives 10",
            2: "python train.py --dir_name log/num_prim_25 --nb_primitives 25",
        }
        self.data_augmentation = {
            1: "python train.py --dir_name log/data_augmentation_1 --nb_primitives 10 --random_translation 1",
            2: "python train.py --dir_name log/data_augmentation_2 --nb_primitives 10 --random_translation 1 --anisotropic_scaling 1",
            3: "python train.py --dir_name log/data_augmentation_3 --nb_primitives 10 --data_augmentation_axis_rotation 1 --data_augmentation_random_flips 1 --random_translation 1 --anisotropic_scaling 1",
            4: "python train.py --dir_name log/data_augmentation_4 --nb_primitives 10 --random_rotation 1 --data_augmentation_random_flips 1 --random_translation 1 --anisotropic_scaling 1",
        }

        self.number_points = {
            1: "python train.py --dir_name log/number_points_10000 --nb_primitives 10 --number_points 10000",
            2: "python train.py --dir_name log/number_points_1000 --nb_primitives 10 --number_points 1000",
        }

        self.normalization = {
            1: "python train.py --dir_name log/normalization_boundingBox --nb_primitives 10 --normalization BoundingBox",
            2: "python train.py --dir_name log/normalization_identity --nb_primitives 10 --normalization Identity",
            3: "python train.py --dir_name log/normalization_unitBall --nb_primitives 10 --normalization UnitBall",
        }

        self.bottleneck_size = {
            1: "python train.py --dir_name log/bottleneck_size_128 --nb_primitives 10 --bottleneck_size 128",
            2: "python train.py --dir_name log/bottleneck_size_2048 --nb_primitives 10 --bottleneck_size 2048",
        }

        self.multi_gpu = {
            1: "python train.py --dir_name log/multi_gpu_1 --multi_gpu 0 1 2 3 --batch_size 128",
            2: "python train.py --dir_name log/multi_gpu_10 --multi_gpu 0 1 2 3 --nb_primitives 10 --batch_size 128",
        }

        self.activation = {
            1: "python train.py --dir_name log/activation_sigmoid --nb_primitives 10  --activation sigmoid",
            2: "python train.py --dir_name log/activation_softplus --nb_primitives 10  --activation softplus",
            3: "python train.py --dir_name log/activation_logsigmoid --nb_primitives 10  --activation logsigmoid",
            4: "python train.py --dir_name log/activation_softsign --nb_primitives 10  --activation softsign",
            5: "python train.py --dir_name log/activation_tanh --nb_primitives 10  --activation tanh",
        }

        self.num_layers = {
            1: "python train.py --dir_name log/num_layers_2 --nb_primitives 10  --num_layers 2",
            2: "python train.py --dir_name log/num_layers_3 --nb_primitives 10  --num_layers 3",
            3: "python train.py --dir_name log/num_layers_4 --nb_primitives 10  --num_layers 4",
            4: "python train.py --dir_name log/num_layers_5 --nb_primitives 10  --num_layers 5",
        }

        self.hidden_neurons = {
            1: "python train.py --dir_name log/hidden_neurons_256 --nb_primitives 10  --hidden_neurons 256",
            2: "python train.py --dir_name log/hidden_neurons_128 --nb_primitives 10  --hidden_neurons 128",
            3: "python train.py --dir_name log/hidden_neurons_64 --nb_primitives 10  --hidden_neurons 64",
        }

        self.decoder_type = {
            1: "python train.py --dir_name log/decoder_type --nb_primitives 10  --decoder_type AtlasNet",
            2: "python train.py --dir_name log/decoder_type_light --nb_primitives 10  --decoder_type AtlasNetLight",
        }


exp = Experiments()


def get_first_available_gpu():
    """
    Check if a gpu is free and returns it
    :return: gpu_id
    """
    query = gpustat.new_query()
    for gpu_id in range(len(query)):
        gpu = query[gpu_id]
        print(gpu_id, gpu.memory_used)
        if gpu.memory_used < 2000:
            if gpu.utilization == 0 and gpu.memory_used < 12 and gpu_id == 0 and gpu.processes.__len__() == 0:
                os.system(f"tmux kill-session -t GPU{gpu_id}")
            has = os.system(f"tmux has-session -t GPU{gpu_id} 2>/dev/null")
            if not int(has) == 0:
                return gpu_id
    return -1


def job_scheduler(dict_of_jobs):
    """
    Launch Tmux session each time it finds a free gpu
    :param dict_of_jobs:
    """
    keys = list(dict_of_jobs.keys())
    while len(keys) > 0:
        job_key = keys.pop()
        job = dict_of_jobs[job_key]
        while get_first_available_gpu() < 0:
            print("Waiting to find a GPU for ", job)
            time.sleep(30)  # Sleeps for 30 sec

        gpu_id = get_first_available_gpu()
        name_tmux = f"GPU{gpu_id}"
        cmd = f"conda activate python3;  {job} --multi_gpu {gpu_id} 2>&1 | tee  log_terminals/{gpu_id}_{job_key}.txt; tmux kill-session -t {name_tmux}"
        CMD = f'tmux new-session -d -s {name_tmux} \; send-keys "{cmd}" Enter'
        print(CMD)
        os.system(CMD)
        time.sleep(60)  # Sleeps for 30 sec


def job_scheduler_multi(dict_of_jobs):
    """
    Launch Tmux session each time it finds a free gpu
    :param dict_of_jobs:
    """
    keys = list(dict_of_jobs.keys())
    while len(keys) > 0:
        job_key = keys.pop()
        job = dict_of_jobs[job_key]
        while get_first_available_gpu() < 0:
            print("Waiting to find a GPU for ", job)
            time.sleep(30)  # Sleeps for 30 sec

        gpu_id = get_first_available_gpu()
        name_tmux = f"GPU{gpu_id}"
        cmd = f"conda activate python3;  {job}  2>&1 | tee  log_terminals/{gpu_id}_{job_key}.txt; tmux kill-session -t {name_tmux}"
        CMD = f'tmux new-session -d -s {name_tmux} \; send-keys "{cmd}" Enter'
        print(CMD)
        os.system(CMD)
        time.sleep(60)  # Sleeps for 30 sec


for path in ["log_terminals", "log"]:
    if not os.path.exists(path):
        print(f"Creating {path} folder")
        os.mkdir(path)

job_scheduler(exp.decoder_type)
