from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import sys
sys.path.append('./auxiliary/')
from dataset import *
from model import *
from utils import *
from ply import *
import torch.nn.functional as F
import sys
from tqdm import tqdm
import os
import json
import time, datetime
import subprocess
import pandas as pd
try:
    from script.normalize_obj import *
except:
    print('couldnt load normalize obj')
sys.path.append("./nndistance/")
from modules.nnd import NNDModule
distChamfer =  NNDModule()


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
parser.add_argument('--model', type=str, default = 'trained_models/svr_baseline.pth',  help='yuor path to the trained model')
parser.add_argument('--num_points', type=int, default = 2500,  help='number of points fed to poitnet')

opt = parser.parse_args()
print (opt)

blue = lambda x:'\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)


dataset_test = ShapeNet( SVR=True, normal = False, class_choice = None, train=False)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                          shuffle=False, num_workers=int(opt.workers))

print('testing set', len(dataset_test.datapath))
cudnn.benchmark = True
len_dataset = len(dataset_test)

network = SVR_Baseline(num_points = opt.num_points)
network.cuda()

network.apply(weights_init)
if opt.model != '':
    network.load_state_dict(torch.load(opt.model))
    print("previous weight loaded")

print(network)


train_loss = AverageValueMeter()
val_loss = AverageValueMeter()
metro_PMA_loss = AverageValueMeter()

network.eval()

#reset meters
val_loss.reset()
metro_PMA_loss.reset()
for item in dataset_test.cat:
    dataset_test.perCatValueMeter[item].reset()

results = dataset_test.cat.copy()
for i in results:
    results[i] = 0


#Iterate on the data
for i, data in enumerate(dataloader_test, 0):
    img, points, cat , objpath, fn = data
    cat = cat[0]
    fn = fn[0]
    img = Variable(img)
    img = img.cuda()
    results[cat] = results[cat] + 1
    points = Variable(points)
    points = points.transpose(2,1).contiguous()
    points = points.cuda()
    pointsReconstructed  = network.forward(img)
    dist1, dist2 = distChamfer(points.transpose(2,1).contiguous(), pointsReconstructed)
    loss_net = ((torch.mean(dist1) + torch.mean(dist2)))
    val_loss.update(loss_net.data[0])
    dataset_test.perCatValueMeter[cat].update(loss_net.data[0])

    if results[cat] > 20:
        #only save files for 20 objects per category
        continue
    print(results)

    if not os.path.exists(opt.model[:-4]):
        os.mkdir(opt.model[:-4])
        print('created dir', opt.model[:-4])

    if not os.path.exists(opt.model[:-4] + "/" + str(dataset_test.cat[cat])):
        os.mkdir(opt.model[:-4] + "/" + str(dataset_test.cat[cat]))
        print('created dir', opt.model[:-4] + "/" + str(dataset_test.cat[cat]))

    write_ply(filename=opt.model[:-4] + "/" + str(dataset_test.cat[cat]) + "/" + fn+"_GT", points=pd.DataFrame(points.transpose(2,1).contiguous().cpu().data.squeeze().numpy()), as_text=True)
    write_ply(filename=opt.model[:-4] + "/" + str(dataset_test.cat[cat]) + "/" + fn+"_gen", points=pd.DataFrame(pointsReconstructed.cpu().data.squeeze().numpy()), as_text=True)


log_table = {
  "metro_PMA_loss" : metro_PMA_loss.avg,
  "val_loss" : val_loss.avg,
}
for item in dataset_test.cat:
    print(item, dataset_test.perCatValueMeter[item].avg)
    log_table.update({item: dataset_test.perCatValueMeter[item].avg})
print(log_table)

with open('stats.txt', 'a') as f: #open and append
    f.write('json_stats: ' + json.dumps(log_table) + '\n')
