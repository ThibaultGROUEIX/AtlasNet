from __future__ import print_function
import argparse
import random
import torch
import sys
sys.path.append('./auxiliary/')
from dataset import *
from model import *
from utils import *
from ply import *
import os
import json
import pandas as pd

# =============PARAMETERS======================================== #
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
parser.add_argument('--nepoch', type=int, default=50000, help='number of epochs to train for')
parser.add_argument('--model', type=str, default = 'trained_models/ae_baseline.pth',  help='yuor path to the trained model')
parser.add_argument('--num_points', type=int, default = 2500,  help='number of points fed to poitnet')

opt = parser.parse_args()
print (opt)
# ========================================================== #



# =============DEFINE CHAMFER LOSS======================================== #
def pairwise_dist(x, y):
    xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    P = (rx.t() + ry - 2*zz)
    return P


def NN_loss(x, y, dim=0):
    dist = pairwise_dist(x, y)
    values, indices = dist.min(dim=dim)
    return values.mean()


def distChamfer(a,b):
    x,y = a,b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2,1))
    yy = torch.bmm(y, y.transpose(2,1))
    zz = torch.bmm(x, y.transpose(2,1))
    diag_ind = torch.arange(0, num_points).type(torch.cuda.LongTensor)
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2,1) + ry - 2*zz)
    return P.min(1)[0], P.min(2)[0]
# ========================================================== #

blue = lambda x:'\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)


# ===================CREATE DATASET================================= #
dataset_test = ShapeNet( normal = False, class_choice = None, train=False)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                          shuffle=False, num_workers=int(opt.workers))

print('testing set', len(dataset_test.datapath))
len_dataset = len(dataset_test)
# ========================================================== #

# ===================CREATE network================================= #
network = AE_Baseline(num_points = opt.num_points)
network.cuda()

network.apply(weights_init)
if opt.model != '':
    network.load_state_dict(torch.load(opt.model))
    print("previous weight loaded")

print(network)
network.eval()
# ========================================================== #

# =============DEFINE stuff for logs ======================================== #
train_loss = AverageValueMeter()
val_loss = AverageValueMeter()
# ========================================================== #


#reset meters
val_loss.reset()
for item in dataset_test.cat:
    dataset_test.perCatValueMeter[item].reset()

results = dataset_test.cat.copy()
for i in results:
    results[i] = 0


# =============TESTING LOOP======================================== #
#Iterate on the data
with torch.no_grad():
    for i, data in enumerate(dataloader_test, 0):
        _, points, cat , objpath, fn = data
        cat = cat[0]
        fn = fn[0]
        results[cat] = results[cat] + 1
        points = points.transpose(2,1).contiguous()
        points = points.cuda()
        pointsReconstructed  = network.forward(points)
        dist1, dist2 = distChamfer(points.transpose(2,1).contiguous(), pointsReconstructed)
        loss_net = ((torch.mean(dist1) + torch.mean(dist2)))
        val_loss.update(loss_net.item())
        dataset_test.perCatValueMeter[cat].update(loss_net.item())

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
      "val_loss" : val_loss.avg,
    }
    for item in dataset_test.cat:
        print(item, dataset_test.perCatValueMeter[item].avg)
        log_table.update({item: dataset_test.perCatValueMeter[item].avg})
    print(log_table)

    with open('stats.txt', 'a') as f: #open and append
        f.write('json_stats: ' + json.dumps(log_table) + '\n')
