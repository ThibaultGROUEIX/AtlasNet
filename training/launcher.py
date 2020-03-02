import os
import gpustat
import time

"""
    Author : Thibault Groueix 01.11.2019
"""

class Experiments(object):
    def __init__(self):
        self.conformal = {
            # 1: "python train.py --dir_name log/atlasnet_0_1 --activation softplus --template_type SQUARE --nb_primitives 5 --batch_size 32 --number_points 2048 --conformal_regul --lambda_conformal_regul 0.1",
            # 1: "python train.py --dir_name log/atlasnet_1_0_with_2 --activation softplus --template_type SQUARE --nb_primitives 1 --batch_size 32 --number_points 2048 --conformal_regul --lambda_conformal_regul 1",
            # 2: "python train.py --dir_name log/atlasnet_1_0_without_2 --activation softplus --template_type SQUARE --nb_primitives 1 --batch_size 32 --number_points 2048  --lambda_conformal_regul 0",
            3: "python train.py --dir_name log/atlasnet_1_0_with_3 --activation softplus --template_type SQUARE --nb_primitives 1 --batch_size 32 --number_points 2048  --conformal_regul --lambda_conformal_regul 0.001",
            # 4: "python train.py --dir_name log/atlasnet_0_01 --activation softplus --template_type SQUARE --nb_primitives 25 --batch_size 32 --number_points 2048 --conformal_regul --lambda_conformal_regul 0.01",
            # 5: "python train.py --dir_name log/atlasnet_0_1_noBN --activation softplus --template_type SQUARE --nb_primitives 25 --batch_size 32 --number_points 2048 --conformal_regul --lambda_conformal_regul 0.1 --remove_all_batchNorms"
        }

        self.atlasnet = {
            1: "python train.py --shapenet13 --dir_name log/atlasnet_autoencoder_1_sphere  --nb_primitives 1 --template_type SPHERE",
            2: "python train.py --shapenet13 --dir_name log/atlasnet_autoencoder_25_squares  --nb_primitives 25 --template_type SQUARE",
            3: "python train.py --shapenet13 --dir_name log/atlasnet_singleview_1_sphere_tmp_True --nb_primitives 1 --template_type SPHERE --SVR --reload_decoder_path log/atlasnet_autoencoder_1_sphere/network.pth --train_only_encoder",
            4: "python train.py --shapenet13 --dir_name log/atlasnet_singleview_25_squares_tmp_True  --nb_primitives 25 --template_type SQUARE  --SVR  --reload_decoder_path log/atlasnet_autoencoder_25_squares/network.pth --train_only_encoder",
            5: "python train.py --shapenet13 --dir_name log/atlasnet_singleview_1_sphere --nb_primitives 1 --template_type SPHERE --SVR --reload_model_path log/atlasnet_singleview_1_sphere_tmp_True/network.pth --lrate 0.0001 --nepoch 50 --lr_decay_1 40",
            6: "python train.py --shapenet13 --dir_name log/atlasnet_singleview_25_squares  --nb_primitives 25 --template_type SQUARE  --SVR  --reload_model_path log/atlasnet_singleview_25_squares_tmp_True/network.pth  --lrate 0.0001 --nepoch 50 --lr_decay_1 40",
        }

        self.atlasnet_test = {
            1: "python train.py --shapenet13 --dir_name training/trained_models/atlasnet_autoencoder_1_sphere  --nb_primitives 1 --template_type SPHERE --run_single_eval --no_metro",
            2: "python train.py --shapenet13 --dir_name training/trained_models/atlasnet_autoencoder_25_squares  --nb_primitives 25 --template_type SQUARE  --run_single_eval --no_metro",
            3: "python train.py --shapenet13 --dir_name training/trained_models/atlasnet_singleview_1_sphere  --nb_primitives 1 --template_type SPHERE --run_single_eval --no_metro  --SVR",
            4: "python train.py --shapenet13 --dir_name training/trained_models/atlasnet_singleview_25_squares  --nb_primitives 25 --template_type SQUARE  --run_single_eval --no_metro  --SVR",
        }
        self.template = {
            1: "python train.py --shapenet13 --dir_name log/template_sphere --template_type SPHERE",
            2: "python train.py --shapenet13 --dir_name log/template_square --template_type SQUARE --nb_primitives 1",
        }
        self.num_prim = {
            1: "python train.py --shapenet13 --dir_name log/num_prim_10 --nb_primitives 10  --template_type SQUARE",
            2: "python train.py --shapenet13 --dir_name log/num_prim_25 --nb_primitives 25  --template_type SQUARE",
        }
        self.data_augmentation = {
            1: "python train.py --shapenet13 --dir_name log/data_augmentation_1 --nb_primitives 10 --random_translation 1",
            2: "python train.py --shapenet13 --dir_name log/data_augmentation_2 --nb_primitives 10 --random_translation 1 --anisotropic_scaling 1",
            3: "python train.py --shapenet13 --dir_name log/data_augmentation_3 --nb_primitives 10 --data_augmentation_axis_rotation 1 --data_augmentation_random_flips 1 --random_translation 1 --anisotropic_scaling 1",
            4: "python train.py --shapenet13 --dir_name log/data_augmentation_4 --nb_primitives 10 --random_rotation 1 --data_augmentation_random_flips 1 --random_translation 1 --anisotropic_scaling 1",
        }

        self.number_points = {
            1: "python train.py --shapenet13 --dir_name log/number_points_8000 --nb_primitives 10 --number_points 8000",
            2: "python train.py --shapenet13 --dir_name log/number_points_1000 --nb_primitives 10 --number_points 1000",
        }

        self.normalization = {
            1: "python train.py --shapenet13 --dir_name log/normalization_boundingBox --nb_primitives 10 --normalization BoundingBox",
            2: "python train.py --shapenet13 --dir_name log/normalization_identity --nb_primitives 10 --normalization Identity",
            3: "python train.py --shapenet13 --dir_name log/normalization_unitBall --nb_primitives 10 --normalization UnitBall",
        }

        self.bottleneck_size = {
            1: "python train.py --shapenet13 --dir_name log/bottleneck_size_128 --nb_primitives 10 --bottleneck_size 128",
            2: "python train.py --shapenet13 --dir_name log/bottleneck_size_2048 --nb_primitives 10 --bottleneck_size 2048",
            3: "python train.py --shapenet13 --dir_name log/bottleneck_size_4096 --nb_primitives 10 --bottleneck_size 4096",
        }

        self.multi_gpu = {
            1: "python train.py --shapenet13 --dir_name log/multi_gpu_1 --multi_gpu 0 1 2 3 --batch_size 128",
            2: "python train.py --shapenet13 --dir_name log/multi_gpu_10 --multi_gpu 0 1 2 3 --nb_primitives 10 --batch_size 128",
        }

        self.activation = {
            1: "python train.py --shapenet13 --dir_name log/activation_sigmoid --nb_primitives 10  --activation sigmoid",
            2: "python train.py --shapenet13 --dir_name log/activation_softplus --nb_primitives 10  --activation softplus",
            3: "python train.py --shapenet13 --dir_name log/activation_logsigmoid --nb_primitives 10  --activation logsigmoid",
            4: "python train.py --shapenet13 --dir_name log/activation_softsign --nb_primitives 10  --activation softsign",
            5: "python train.py --shapenet13 --dir_name log/activation_tanh --nb_primitives 10  --activation tanh",
        }

        self.num_layers = {
            1: "python train.py --shapenet13 --dir_name log/num_layers_2 --nb_primitives 10  --num_layers 2",
            2: "python train.py --shapenet13 --dir_name log/num_layers_3 --nb_primitives 10  --num_layers 3",
            3: "python train.py --shapenet13 --dir_name log/num_layers_4 --nb_primitives 10  --num_layers 4",
            4: "python train.py --shapenet13 --dir_name log/num_layers_5 --nb_primitives 10  --num_layers 5",
        }

        self.hidden_neurons = {
            1: "python train.py --shapenet13 --dir_name log/hidden_neurons_256 --nb_primitives 10  --hidden_neurons 256",
            2: "python train.py --shapenet13 --dir_name log/hidden_neurons_128 --nb_primitives 10  --hidden_neurons 128",
            3: "python train.py --shapenet13 --dir_name log/hidden_neurons_64 --nb_primitives 10  --hidden_neurons 64",
            4: "python train.py --shapenet13 --dir_name log/hidden_neurons_512 --nb_primitives 10  --hidden_neurons 512",
            5: "python train.py --shapenet13 --dir_name log/hidden_neurons_1024 --nb_primitives 10  --hidden_neurons 1024",
        }

        self.single_view = {
            1: "python train.py --dir_name log/single_view --shapenet13 --nb_primitives 10  --SVR",
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


def job_scheduler_parralel(dict_of_jobs):
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
            time.sleep(15)  # Sleeps for 30 sec

        gpu_id = get_first_available_gpu()
        name_tmux = f"GPU{gpu_id}"
        cmd = f"conda activate python3;  {job} --multi_gpu {gpu_id} 2>&1 | tee  log_terminals/{gpu_id}_{job_key}.txt; tmux kill-session -t {name_tmux}"
        CMD = f'tmux new-session -d -s {name_tmux} \; send-keys "{cmd}" Enter'
        print(CMD)
        os.system(CMD)
        time.sleep(15)  # Sleeps for 30 sec


def job_scheduler_sequential(dict_of_jobs):
    """
    Choose a gpum then launches jobs sequentially on that GPU in tmux sessions.
    :param dict_of_jobs:
    """
    keys = list(dict_of_jobs.keys())
    while get_first_available_gpu() < 0:
        time.sleep(15)  # Sleeps for 30 sec

    gpu_id = get_first_available_gpu()

    while len(keys) > 0:
        has = os.system(f"tmux has-session -t GPU{gpu_id} 2>/dev/null")
        if not int(has) == 0:
            job_key = keys.pop()
            job = dict_of_jobs[job_key]
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

job_scheduler_parralel(exp.conformal)

