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
        self.training = {
            1: "python training/train.py --dir_name log/10prim_1 --nb_primitives 10 --random_rotation 0 --data_augmentation_axis_rotation 0 --data_augmentation_random_flips 1 --random_translation 1 --anisotropic_scaling 0",
            2: "python training/train.py --dir_name log/10prim_2 --nb_primitives 10 --random_rotation 0 --data_augmentation_axis_rotation 0 --data_augmentation_random_flips 1 --random_translation 1 --anisotropic_scaling 1",
            3: "python training/train.py --dir_name log/10prim_3 --nb_primitives 10 --random_rotation 0 --data_augmentation_axis_rotation 1 --data_augmentation_random_flips 1 --random_translation 1 --anisotropic_scaling 1",
            4: "python training/train.py --dir_name log/10prim_4 --nb_primitives 10 --random_rotation 1 --data_augmentation_axis_rotation 0 --data_augmentation_random_flips 1 --random_translation 1 --anisotropic_scaling 1",
            5: "python training/train.py --dir_name log/10prim_points --nb_primitives 10 --number_points 10000",
            6: "python training/train.py --dir_name log/10prim --nb_primitives 10",
            7: "python training/train.py --dir_name log/1prim --nb_primitives 1",
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
            if gpu.utilization == 0 and gpu.memory_used<12 and gpu_id==0 and gpu.processes.__len__()==0:
                os.system(f"tmux kill-session -t GPU{gpu_id}")
            has = os.system(f"tmux has-session -t GPU{gpu_id} 2>/dev/null")
            if not int(has)==0:
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
            time.sleep(30) # Sleeps for 30 sec

        gpu_id = get_first_available_gpu()
        name_tmux = f"GPU{gpu_id}"
        cmd = f"conda activate python3;  {job} --multi_gpu {gpu_id} 2>&1 | tee  log_terminals/{gpu_id}_{job_key}.txt; tmux kill-session -t {name_tmux}"
        CMD = f'tmux new-session -d -s {name_tmux} \; send-keys "{cmd}" Enter'
        print(CMD)
        os.system(CMD)
        time.sleep(60)  # Sleeps for 30 sec


for path in ["log_terminals", "log"]:
    if not os.path.exists(path):
        print(f"Creating {path} folder")
        os.mkdir(path)


job_scheduler(exp.training)
