#/bin/sh
gpu_id=0
env=AE_AtlasNet
nb_primitives=25
source activate pytorch-sources
. switch-gpu $gpu_id
git pull
python ./training/train_AE_AtlasNet.py --env $env  |& tee ${env}.txt
FreeSms.py "code finished or bugged ${env} on ${HOSTNAME}"
