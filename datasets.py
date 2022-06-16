import torch
#from torchvision import transforms
import glob
from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SEMIDataset(Dataset):
    def __init__(self, type='train', numb=None):
        super(SEMIDataset, self).__init__()
        self.type = type
        self.numb = numb
#         self.transform=transform
       
        if self.type=='train':
            self.sem_path = './Train/SEM/'
            self.depth_path = './Train/Depth/'
        elif self.type =='val':
            self.sem_path = './Validation/SEM/'
            self.depth_path = './Validation/Depth/'
        
#         self.sem_list=[x for x in os.listdir(self.sem_path) if x[-5]==str(self.numb)]
        self.depth_list = os.listdir(self.depth_path)

    def __len__(self):
        return len(self.depth_list)

    def __getitem__(self, idx):
#         sem_img_path=self.sem_list[idx]
        depth_img_path = self.depth_list[idx]
        sem_img_path = self.sem_path + depth_img_path[:-4]
        
        if self.numb == None:
            img = []
            for i in range(4):
                img.append(torch.tensor([np.array(Image.open(sem_img_path + '_itr{}.png'.format(i)))]))
            img = torch.stack(img).float()
        else:
            img = torch.tensor([np.array(Image.open(sem_img_path + '_itr{}.png'.format(self.numb)))], dtype=torch.float32)
            
        gt = torch.tensor([np.array(Image.open(self.depth_path+depth_img_path))], dtype=torch.float32)
        
#         if self.transform is not None:
#             img=self.transform(Image.open(sem_img_path))
#             gt = self.transform(Image.open(depth_img_path))

        return img / 255, gt / 255
    

class TestSEMIDataset(Dataset):
    #기존에 numb였던게 의미가 없어지면서, type으로 대체함. Test, Train, Validation
    def __init__(self, type='Test'):
        super().__init__()
        self.type=type
#         self.transform=transform 

        self.sem_path = './{}/'.format(self.type)+'SEM/'
        self.sem_list = [x for x in os.listdir(self.sem_path)] #if x[-5]==self.numb] if self.numb != None else [x for x in os.listdir(self.sem_path) if x[-5]==0]
    
    def __len__(self):
        return len(self.sem_list)
    
    def __getitem__(self, idx):
        sem_img_path=self.sem_path + self.sem_list[idx]

#         if self.transform is not None:
#             img=self.transform(Image.open(sem_img_path))
        
        '''if self.numb == None:
            img = [torch.tensor([np.array(Image.open(sem_img_path))])]
            for i in range(1, 4):
                img.append(torch.tensor([np.array(Image.open(sem_img_path[:-5] + '{}.png'.format(i)))]))
            img = torch.stack(img).float()
        else:'''
        img = torch.tensor([np.array(Image.open(sem_img_path))]).float()

        return img / 255, self.sem_list[idx] #파일이름을 얻기 위함.