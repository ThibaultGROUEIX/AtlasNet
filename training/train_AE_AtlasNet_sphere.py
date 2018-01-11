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
sys.path.append('./aux/')
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
import visdom

sys.path.append("./nndistance/")
from modules.nnd import NNDModule
distChamfer =  NNDModule()


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
parser.add_argument('--nepoch', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--model', type=str, default = '',  help='optional reload model path')
parser.add_argument('--num_points', type=int, default = 2500,  help='number of points')
parser.add_argument('--nb_primitives', type=int, default = 1,  help='number of primitives in the atlas')
parser.add_argument('--super_points', type=int, default = 2500,  help='number of input points to pointNet, not used by default')
parser.add_argument('--env', type=str, default ="main"   ,  help='visdom environment')

opt = parser.parse_args()
print (opt)

#Launch visdom for visualization
vis = visdom.Visdom(port = 8888, env=opt.env)
now = datetime.datetime.now()
save_path = now.isoformat()
dir_name =  os.path.join('log', save_path)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
logname = os.path.join(dir_name, 'log.txt')

blue = lambda x:'\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
best_val_loss = 10


#Create train/test dataloader
dataset = ShapeNet( normal = False, class_choice = None, train=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))
dataset_test = ShapeNet( normal = False, class_choice = None, train=False)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize,
                                          shuffle=False, num_workers=int(opt.workers))


print('training set', len(dataset.datapath))
print('testing set', len(dataset_test.datapath))

cudnn.benchmark = True
len_dataset = len(dataset)

#create network
network = AE_AtlasNet_SPHERE(num_points = opt.num_points, nb_primitives = opt.nb_primitives)
network.cuda() #put network on GPU

# print(network)

network.apply(weights_init) #initialization of the weight

if opt.model != '':
    network.load_state_dict(torch.load(opt.model))
    print(" Previous weight loaded ")

lrate = 0.001 #learning rate
optimizer = optim.Adam(network.parameters(), lr = lrate)

#meters to record stats on learning
train_loss = AverageValueMeter()
val_loss = AverageValueMeter()
with open(logname, 'a') as f: #open and append
        f.write(str(network) + '\n')

#initialize learning curve on visdom, and color for each primitive in visdom display
win_curve = vis.line(
    X = np.array( [0] ),
    Y = np.array( [0] ),
)
val_curve = vis.line(
    X = np.array( [0] ),
    Y = np.array( [1] ),
)
labels_generated_points = torch.Tensor(range(1, (opt.nb_primitives+1)*(opt.num_points/opt.nb_primitives)+1)).view(opt.num_points/opt.nb_primitives,(opt.nb_primitives+1)).transpose(0,1)
labels_generated_points = (labels_generated_points)%(opt.nb_primitives+1)
labels_generated_points = labels_generated_points.contiguous().view(-1)
print(labels_generated_points)

#start of the learning loop
for epoch in range(opt.nepoch):
    #TRAIN MODE
    train_loss.reset()
    network.train()
    for i, data in enumerate(dataloader, 0):
        optimizer.zero_grad()
        img, points, cat, _, _ = data
        points = Variable(points)
        points = points.transpose(2,1).contiguous()
        points = points.cuda()
        #SUPER_RESOLUTION optionally reduce the size of the points fed to PointNet
        # points = points[:,:,:opt.super_points].contiguous()
        #END SUPER RESOLUTION
        pointsReconstructed  = network(points) #forward pass
        dist1, dist2 = distChamfer(points.transpose(2,1).contiguous(), pointsReconstructed) #loss function
        loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
        loss_net.backward()
        train_loss.update(loss_net.data[0])
        optimizer.step() #gradient update

        # VIZUALIZE
        if i%50 <= 0:
            vis.scatter(X = points.transpose(2,1).contiguous()[0].data.cpu(),
                    win = 'REAL_TRAIN',
                    opts = dict(
                        title = "REAL_TRAIN",
                        markersize = 2,
                        ),
                    )
            vis.scatter(X = pointsReconstructed[0].data.cpu(),
                    Y = labels_generated_points[0:pointsReconstructed.size(1)],
                    win = 'FAKE_TRAIN',
                    opts = dict(
                        title="FAKE_TRAIN",
                        markersize=2,
                        ),
                    )

        print('[%d: %d/%d] train loss:  %f ' %(epoch, i, len_dataset/32, loss_net.data[0]))

    #UPDATE CURVES
    if train_loss.avg != 0:
        vis.updateTrace(
            X = np.array([epoch]),
            Y = np.log(np.array([train_loss.avg])),
            win = win_curve,
            name = 'Chamfer train'
        )

    # VALIDATION
    val_loss.reset()
    for item in dataset_test.cat:
        dataset_test.perCatValueMeter[item].reset()

    network.eval()
    for i, data in enumerate(dataloader_test, 0):
        img, points, cat, _ , _ = data
        points = Variable(points, volatile=True)
        points = points.transpose(2,1).contiguous()
        points = points.cuda()
        #SUPER_RESOLUTION
        # points = points[:,:,:opt.super_points].contiguous()
        #END SUPER RESOLUTION
        pointsReconstructed  = network(points)
        dist1, dist2 = distChamfer(points.transpose(2,1).contiguous(), pointsReconstructed)
        loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
        val_loss.update(loss_net.data[0])
        dataset_test.perCatValueMeter[cat[0]].update(loss_net.data[0])
        if i%200 ==0 :
            vis.scatter(X = points.transpose(2,1).contiguous()[0].data.cpu(),
                    win = 'REAL_CHAIR',
                    opts = dict(
                        title = "REAL_CHAIR",
                        markersize = 2,
                        ),
                    )
            vis.scatter(X = pointsReconstructed[0].data.cpu(),
                    Y = labels_generated_points[0:pointsReconstructed.size(1)],
                    win = 'FAKE_CHAIR',
                    opts = dict(
                        title = "FAKE_CHAIR",
                        markersize = 2,
                        ),
                    )
        print('[%d: %d/%d] val loss:  %f ' %(epoch, i, len(dataset_test), loss_net.data[0]))

    #UPDATE CURVES
    if val_loss.avg != 0:
        vis.updateTrace(
            X = np.array([epoch]),
            Y = np.log(np.array([val_loss.avg])),
            win = val_curve,
            name = 'Chamfer val'
        )

    #dump stats in log file
    log_table = {
      "train_loss" : train_loss.avg,
      "val_loss" : val_loss.avg,
      "epoch" : epoch,
      "lr" : lrate,
      "super_points" : opt.super_points,
      "bestval" : best_val_loss,

    }
    print(log_table)
    for item in dataset_test.cat:
        print(item, dataset_test.perCatValueMeter[item].avg)
        log_table.update({item: dataset_test.perCatValueMeter[item].avg})
    with open(logname, 'a') as f: #open and append
        f.write('json_stats: ' + json.dumps(log_table) + '\n')

    #save best network
    if best_val_loss > val_loss.avg:
        best_val_loss = val_loss.avg
        print('New best loss : ', best_val_loss)
        print('saving net...')
        torch.save(network.state_dict(), '%s/network.pth' % (dir_name))
