import argparse
import os
import copy
import sys
sys.path.append(os.getcwd())

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader

from tqdm import tqdm
from timm.scheduler.cosine_lr import CosineLRScheduler

from model import CNNBB
from datasets import SEMIDataset, TestSEMIDataset

def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--dataset_name', type=str, default='train', help= 'val')
parser.add_argument('--sem_id', type=int, default=None)
parser.add_argument('--checkpoint_dir', type=str, default=None)

parser.add_argument('--loss', type=str, default='l1', help='l1/mse/rmse')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_epochs', type=int, default = 300)
parser.add_argument('--d_embed', type=int, default =128)
parser.add_argument('--seed', type=int, default = 133)
args = parser.parse_args()

save_path_checkpoints = './checkpoint/' if args.checkpoint_dir == None else args.checkpoint_dir
# save_path_checkpoints = checkpoint_dir+args.dataset_name
save_path_map = './Infermap'+args.dataset_name

if not os.path.exists(save_path_checkpoints):
    os.mkdir(save_path_checkpoints)
if not os.path.exists('./Infermap'):
    os.mkdir('./Infermap')
if not os.path.exists(save_path_map):
    os.mkdir(save_path_map)
'''
if args.dataset_name == 'Test':
    save_path_map = './Infermap'+args.dataset_name
    if not(os.path.exists('./Infermap')):
        os.mkdir('./Infermap')
    if not(save_path_map):
        os.mkdir(save_path_map)'''

cudnn.benchmark=True
device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')

torch.manual_seed(args.seed)

model = CNNBB(aggregation=True if args.sem_id == None else False, d_embed=args.d_embed).to(device)

if args.loss == 'l1':
    criterion = nn.L1Loss().to(device)
elif args.loss == 'mse':
    criterion = nn.MSELoss().to(device)
elif args.loss == 'rmse':
    criterion = RMSELoss

# optimizer = optim.Adam([
#     {'params': model.conv1.parameters()},
#     {'params': model.conv2.parameters()},
#     {'params': model.conv3.parameters(), 'lr': args.lr * 0.1}
# ], lr=args.lr)
optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=1e-4)

dataset = SEMIDataset('train', args.sem_id)
dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4, pin_memory=False, drop_last=False
                        )
size = len(dataloader.dataset)

valid_dataset = SEMIDataset('val', args.sem_id)
valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=False, drop_last=False
                              )
valid_size = len(valid_dataloader.dataset)

max_iter = (size // args.batch_size + 1) * args.num_epochs

# scheduler : warmup + cosine lr decay
scheduler = CosineLRScheduler(optimizer,
                              t_initial=max_iter,
                              lr_min=args.lr*0.01,
                              warmup_lr_init=args.lr*0.001,
                              warmup_t=max_iter//10,
                              cycle_limit=1,
                              t_in_epochs=False,
                             )

min_valid_rmse = 1000
min_valid_epoch = 0
_iter = 0

for epoch in range(args.num_epochs):
    model.train()
    
    with tqdm(len(dataset)) as t:
        t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs-1))
        
        for batch, (SEM, DEPTH) in enumerate(dataloader):
            n_batch = SEM.shape[0]
            
            SEM = SEM.view(n_batch*4, *SEM.shape[2:]).to(device) if args.sem_id == None else SEM.to(device)
#             DEPTH = torch.repeat_interleave(DEPTH, 4, dim=0).to(device)
            DEPTH = DEPTH.to(device)

            pred = model(SEM)
            loss = criterion(pred, DEPTH)

            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            
            optimizer.step()
            
            _iter += 1
            scheduler.step_update(_iter)

            t.update(n_batch)

            if batch % 100 ==0:
                loss, current = loss.item(), batch*n_batch
                tqdm.write(f"loss: {loss:>6f} [{current:>5d}/{size:>5d}]")

    torch.save(model.state_dict(), os.path.join(save_path_checkpoints +'epoch_{}.pt'.format(epoch)))
    
    # Validate
    model.eval()
    mse = 0
    with torch.no_grad():
        for batch, (SEM, DEPTH) in enumerate(valid_dataloader):
            n_batch = SEM.shape[0]
            
            SEM = SEM.view(n_batch*4, *SEM.shape[2:]).to(device) if args.sem_id == None else SEM.to(device)
#             DEPTH = torch.repeat_interleave(DEPTH, 4, dim=0).to(device)
            DEPTH = DEPTH.to(device)

            pred = model(SEM)
            mse += ((pred - DEPTH) ** 2).sum()
        
        rmse = torch.sqrt(mse / (valid_size * SEM.shape[-1] * SEM.shape[-2])).item()
        
        if rmse <= min_valid_rmse:
            min_valid_rmse = rmse
            min_valid_epoch = epoch
            
    print('validation: {}\n'.format(rmse))

print('minimum validation {0} at epoch {1}'.format(min_valid_rmse, min_valid_epoch))
