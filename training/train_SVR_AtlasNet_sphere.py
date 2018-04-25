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
import visdom

sys.path.append("./nndistance/")
from modules.nnd import NNDModule
distChamfer =  NNDModule()

best_val_loss = 10

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
parser.add_argument('--nepoch', type=int, default=400, help='number of epochs to train for')
parser.add_argument('--model_preTrained_AE', type=str, default = 'trained_models/ae_atlasnet_sphere.pth',  help='model path')
parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--num_points', type=int, default = 2500,  help='number of points')
parser.add_argument('--nb_primitives', type=int, default = 1,  help='number of primitives')
parser.add_argument('--env', type=str, default ="main"   ,  help='visdom env')
parser.add_argument('--fix_decoder', type=bool, default = True   ,  help='if set to True, on the the resnet encoder is trained')

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

#Create train/test dataloader on new views and test dataset on new models

dataset = ShapeNet( SVR=True, normal = False, class_choice = None, train=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

dataset_test = ShapeNet( SVR=True, normal = False, class_choice = None, train=False)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize,
                                          shuffle=False, num_workers=int(opt.workers))

dataset_test_view = ShapeNet( SVR=True, normal = False, class_choice = None, train=True, gen_view=True)
dataloader_test_view = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize,
                                          shuffle=False, num_workers=int(opt.workers))


print('training set', len(dataset.datapath))
print('testing set', len(dataset_test.datapath))

cudnn.benchmark = True
len_dataset = len(dataset)

# LOAD Pretrained autoencoder and check its performance on testing set
network_preTrained_autoencoder = AE_AtlasNet_SPHERE(num_points = opt.num_points, nb_primitives = opt.nb_primitives)
network_preTrained_autoencoder.cuda()
network_preTrained_autoencoder.load_state_dict(torch.load(opt.model_preTrained_AE ))
if opt.fix_decoder:
    val_loss = AverageValueMeter()
    val_loss.reset()
    network_preTrained_autoencoder.eval()
    for i, data in enumerate(dataloader_test, 0):
        img, points, cat, _ , _= data
        points = Variable(points, volatile=True)
        points = points.transpose(2,1).contiguous()
        points = points.cuda()
        pointsReconstructed  = network_preTrained_autoencoder(points)
        dist1, dist2 = distChamfer(points.transpose(2,1).contiguous(), pointsReconstructed)
        loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
        val_loss.update(loss_net.data[0])
    print("Previous decoder performances : ", val_loss.avg)

#Create network
network = SVR_AtlasNet_SPHERE(num_points = opt.num_points, nb_primitives = opt.nb_primitives)
network.apply(weights_init)
network.cuda()
if opt.model != '':
    network.load_state_dict(torch.load(opt.model))
    print(" Previous weight loaded ")

if opt.fix_decoder:
    network.decoder = network_preTrained_autoencoder.decoder

print(network)


network_preTrained_autoencoder.cpu()
network.cuda()

lrate = 0.001
params_dict = dict(network.named_parameters())
params = []

if opt.fix_decoder:
    optimizer = optim.Adam(network.encoder.parameters(), lr = lrate)
else:
    optimizer = optim.Adam(network.parameters(), lr = lrate)

num_batch = len(dataset) / opt.batchSize

train_loss = AverageValueMeter()
val_loss = AverageValueMeter()
val_view_loss = AverageValueMeter()
with open(logname, 'a') as f: #open and append
        f.write(str(network) + '\n')


win_curve = vis.line(
    X = np.array( [0] ),
    Y = np.array( [0] ),
)
val_curve = vis.line(
    X = np.array( [0] ),
    Y = np.array( [1] ),
)
val_curve_new_views_same_model = vis.line(
    X = np.array( [0] ),
    Y = np.array( [1] ),
)

trainloss_acc0 = 1e-9
trainloss_accs = 0

labels_generated_points = torch.Tensor(range(1, (opt.nb_primitives+1)*(opt.num_points/opt.nb_primitives)+1)).view(opt.num_points/opt.nb_primitives,(opt.nb_primitives+1)).transpose(0,1)
labels_generated_points = (labels_generated_points)%(opt.nb_primitives+1)
labels_generated_points = labels_generated_points.contiguous().view(-1)
print(labels_generated_points)

for epoch in range(opt.nepoch):
    #TRAIN MODE
    train_loss.reset()
    network.train()
    for i, data in enumerate(dataloader, 0):
        optimizer.zero_grad()

        img, points, cat, _ , _= data
        img = Variable(img)
        img = img.cuda()
        points = Variable(points)
        points = points.cuda()

        pointsReconstructed  = network(img)
        dist1, dist2 = distChamfer(points, pointsReconstructed)
        loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
        trainloss_accs = trainloss_accs * 0.99 + loss_net.data[0]
        trainloss_acc0 = trainloss_acc0 * 0.99 + 1
        loss_net.backward()
        train_loss.update(loss_net.data[0])

        optimizer.step()
        # VIZUALIZE
        if i%50 <= 0:
            vis.image(img[0].data.cpu().contiguous(), win = 'INPUT IMAGE TRAIN',opts = dict( title = "INPUT IMAGE TRAIN"))
            vis.scatter(X = points[0].data.cpu(),
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

        print('[%d: %d/%d] train loss:  %f , %f ' %(epoch, i, len_dataset/32, loss_net.data[0], trainloss_accs/trainloss_acc0))

    #UPDATE CURVES
    if train_loss.avg != 0:
        vis.updateTrace(
            X = np.array([epoch]),
            Y = np.log(np.array([train_loss.avg])),
            win = win_curve,
            name = 'Chamfer train'
        )
    #VALIDATION on same models new views
    if epoch%10==0:
        val_view_loss.reset()
        network.eval()
        for i, data in enumerate(dataloader_test_view, 0):

            img, points, cat, _, _ = data
            img = Variable(img, volatile=True)
            img = img.cuda()
            points = Variable(points, volatile=True)
            points = points.cuda()

            pointsReconstructed  = network(img)
            dist1, dist2 = distChamfer(points, pointsReconstructed)
            loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
            val_view_loss.update(loss_net.data[0])
            dataset_test.perCatValueMeter[cat[0]].update(loss_net.data[0])
            if i%25 ==0 :
                vis.image(img[0].data.cpu().contiguous(), win = 'INPUT IMAGE VAL', opts = dict( title = "INPUT IMAGE TRAIN"))
                vis.scatter(X = points[0].data.cpu(),
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
            # print('[%d: %d/%d] val loss:  %f ' %(epoch, i, len(dataset_test), loss_net.data[0]))

        #UPDATE CURVES
        if val_view_loss.avg != 0:
            vis.updateTrace(
                X = np.array([epoch]),
                Y = np.log(np.array([val_view_loss.avg])),
                win = val_curve_new_views_same_model,
                name = 'Chamfer val new views same models'
            )


    #VALIDATION
    val_loss.reset()
    for item in dataset_test.cat:
        dataset_test.perCatValueMeter[item].reset()

    network.eval()
    for i, data in enumerate(dataloader_test, 0):
        img, points, cat, _, _ = data
        img = Variable(img, volatile=True)
        img = img.cuda()
        points = Variable(points, volatile=True)
        points = points.cuda()

        pointsReconstructed  = network(img)
        dist1, dist2 = distChamfer(points, pointsReconstructed)
        loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
        val_loss.update(loss_net.data[0])
        dataset_test.perCatValueMeter[cat[0]].update(loss_net.data[0])
        if i%25 ==0 :
            vis.image(img[0].data.cpu().contiguous(), win = 'INPUT IMAGE VAL', opts = dict( title = "INPUT IMAGE TRAIN"))
            vis.scatter(X = points[0].data.cpu(),
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



    log_table = {
      "train_loss" : train_loss.avg,
      "val_loss" : val_loss.avg,
      "val_loss_new_views_same_models" : val_view_loss.avg,
      "epoch" : epoch,
      "lr" : lrate,
      "bestval" : best_val_loss,
    }
    print(log_table)
    for item in dataset_test.cat:
        print(item, dataset_test.perCatValueMeter[item].avg)
        log_table.update({item: dataset_test.perCatValueMeter[item].avg})
    with open(logname, 'a') as f: #open and append
        f.write('json_stats: ' + json.dumps(log_table) + '\n')
    if best_val_loss > val_loss.avg:
        best_val_loss = val_loss.avg
        print('New best loss : ', best_val_loss)
        print('saving net...')
        torch.save(network.state_dict(), '%s/network.pth' % (dir_name))
