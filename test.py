import argparse
import os
import copy
import sys
from tokenize import PlainToken
sys.path.append(os.getcwd())

import torch
import torchvision
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from model import CNNBB
from datasets import SEMIDataset, TestSEMIDataset

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--sem_id', type=int, default=0)#
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint')
parser.add_argument('--dataset_name', type=str, default='Test', help= 'Validation, Train')
parser.add_argument('--checkpoint_num', type=int, default=261)

parser.add_argument('--batch_size', type=int, default=64)# 64, 32
parser.add_argument('--d_embed', type=int, default =128)#128, 64
parser.add_argument('--seed', type=int, default = 123)
args = parser.parse_args()

data_load_path = './{}/'.format(args.dataset_name)+'/SEM/'
save_path_map = './Infermap/'+args.dataset_name

if not os.path.exists('./Infermap'):
    os.mkdir('./Infermap')
if not os.path.exists(save_path_map):
    os.mkdir(save_path_map)
    

cudnn.benchmark=True
device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
torch.manual_seed(args.seed)

model = CNNBB(aggregation=True if args.sem_id==None else False, d_embed=args.d_embed)
model.load_state_dict(torch.load(args.checkpoint_dir+'/'+'epoch_{}.pt'.format(args.checkpoint_num), map_location='cpu'))
model=model.to(device)
model.eval()

#type의 경우, 사실 없어도 되는데 혹시 비교하고 싶을 경우를 대비해 dataset_name 에 Test, Validation, Train넣으면 되도록 만들어봄.
dataset=TestSEMIDataset(type=args.dataset_name)
dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=0, pin_memory=False, drop_last=False
                        )


final_length=len(dataloader)
#featuremaps - made from ./Test/SEM, feturemapnames - names of file in ./Test/SEM
for i, (featuremaps, featurenames) in enumerate(dataloader):
    featuremaps =featuremaps.to(device)
    featuremaps = model(featuremaps)
    
    for f_map, f_name in zip(featuremaps, featurenames):
        
        image = torchvision.transforms.ToPILImage()(f_map)
        image.save(save_path_map+'/'+f_name)
        
        #torchvision.utils.save_image(f_map, save_path_map+'/'+f_name)
    
        
#Complete!    
if i+1 == final_length:
    print('Complete!')
                


