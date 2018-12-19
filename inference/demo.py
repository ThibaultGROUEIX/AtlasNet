from __future__ import print_function
import argparse
import random
import numpy as np
from torch.autograd import Variable
import sys
sys.path.append('./auxiliary/')
from dataset import *
from model import *
from utils import *
from ply import *
import pandas as pd


###############################################################
# This script takes as input a 137 * 137 image (from ShapeNet), run it through a trained resnet encoder, then decode it through a trained atlasnet with 25 learned parameterizations, and save the output to output.ply
###############################################################


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default="data/plane_input_demo.png", help='input image')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
parser.add_argument('--model', type=str, default = 'trained_models/svr_atlas_25.pth',  help='your path to the trained model')
parser.add_argument('--num_points', type=int, default = 2500,  help='number of points fed to poitnet')
parser.add_argument('--gen_points', type=int, default = 30000,  help='number of points to generate')
parser.add_argument('--nb_primitives', type=int, default = 25,  help='number of primitives')
parser.add_argument('--cuda', type=int, default = 1,  help='use cuda')

opt = parser.parse_args()
blue = lambda x:'\033[94m' + x + '\033[0m'
opt.manualSeed = random.randint(1, 10000) # fix seed
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

network = SVR_AtlasNet(num_points = opt.num_points, nb_primitives = opt.nb_primitives, cuda = opt.cuda)
if opt.cuda:
    network.cuda()

network.apply(weights_init)
if opt.model != '':
    if opt.cuda:
        network.load_state_dict(torch.load(opt.model))
    else:
        network.load_state_dict(torch.load(opt.model, map_location='cpu'))
    print("previous weight loaded")
    
network.eval()
grain = int(np.sqrt(opt.gen_points/opt.nb_primitives))-1
grain = grain*1.0

#generate regular grid
faces = []
vertices = []
face_colors = []
vertex_colors = []
colors = get_colors(opt.nb_primitives)

for i in range(0,int(grain + 1 )):
        for j in range(0,int(grain + 1 )):
            vertices.append([i/grain,j/grain])

for prim in range(0,opt.nb_primitives):
    for i in range(0,int(grain + 1)):
        for j in range(0,int(grain + 1)):
            vertex_colors.append(colors[prim])

    for i in range(1,int(grain + 1)):
        for j in range(0,(int(grain + 1)-1)):
            faces.append([(grain+1)*(grain+1)*prim + j+(grain+1)*i, (grain+1)*(grain+1)*prim + j+(grain+1)*i + 1, (grain+1)*(grain+1)*prim + j+(grain+1)*(i-1)])
    for i in range(0,(int((grain+1))-1)):
        for j in range(1,int((grain+1))):
            faces.append([(grain+1)*(grain+1)*prim + j+(grain+1)*i, (grain+1)*(grain+1)*prim + j+(grain+1)*i - 1, (grain+1)*(grain+1)*prim + j+(grain+1)*(i+1)])
grid = [vertices for i in range(0,opt.nb_primitives)]
grid_pytorch = torch.Tensor(int(opt.nb_primitives*(grain+1)*(grain+1)),2)
for i in range(opt.nb_primitives):
    for j in range(int((grain+1)*(grain+1))):
        grid_pytorch[int(j + (grain+1)*(grain+1)*i),0] = vertices[j][0]
        grid_pytorch[int(j + (grain+1)*(grain+1)*i),1] = vertices[j][1]


#prepare the input data
from PIL import Image
import torchvision.transforms as transforms

my_transforms = transforms.Compose([
                 transforms.CenterCrop(127),
                 transforms.Resize(size =  224, interpolation = 2),
                 transforms.ToTensor(),
                 # normalize,
            ])


im = Image.open(opt.input)
im = my_transforms(im) #scale
img = im[:3,:,:].unsqueeze(0)



img = Variable(img)
if opt.cuda:
    img = img.cuda()

#forward pass
pointsReconstructed  = network.forward_inference(img, grid)



#Save output 3D model
b = np.zeros((len(faces),4)) + 3
b[:,1:] = np.array(faces)
write_ply(filename=opt.input + str(int(opt.gen_points)), points=pd.DataFrame(torch.cat((pointsReconstructed.cpu().data.squeeze(), grid_pytorch), 1).numpy()), as_text=True, text=True, faces = pd.DataFrame(b.astype(int)))

print("Done demoing! Check out results in data/")
