import numpy as np

import torch
import torch.nn as nn

import copy


# Make clones of a layer.
def clone_layer(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class CNNBB(nn.Module):
    def __init__(self, aggregation=True, num_channels=1, kernel=3, d_embed=64):
        super().__init__()
        self.num_channels=num_channels
        self.aggregation = aggregation
        padding = 1 if kernel == 3 else 0
        
        self.conv1 = nn.Conv2d(self.num_channels, d_embed, kernel_size=kernel, padding=padding)
        self.bn1 = nn.BatchNorm2d(d_embed)
        
        self.conv2 = clone_layer(nn.Conv2d(d_embed, d_embed, kernel_size=kernel, padding=padding), 4)
        self.bn2 = clone_layer(nn.BatchNorm2d(d_embed), 4)
        
        self.conv3 = clone_layer(nn.Conv2d(d_embed, d_embed, kernel_size=kernel, padding=padding), 4)
        self.bn3 = clone_layer(nn.BatchNorm2d(d_embed), 4)
        
        self.conv4 = nn.Conv2d(d_embed, d_embed // 2, kernel_size=kernel, padding=padding)
        self.bn4 = nn.BatchNorm2d(d_embed // 2)
        
        self.conv5 = nn.Conv2d(d_embed // 2, 1, kernel_size=kernel, padding=padding)
        self.relu = nn.ReLU(inplace = True)
        
        if aggregation:
            self.attention_layer = PixelWiseAttention(num_channels, d_embed)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                
    def residual_block(self, convs, bns, x):
        out = convs[0](x)
        out = bns[0](out)
        out = self.relu(out)
        
        out = convs[1](out)
        out = bns[1](out)
        return self.relu(out + x)

    def forward(self, x):
        """
        x : (4*n_batch, channel, height, width) if aggregation == True
            (n_batch, channel, height, width) else
            
        return : (n_batch, channel, height, width)
        """
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)
        
        out2 = self.residual_block(self.conv2[:2], self.bn2[:2], out1)
        out2 = self.residual_block(self.conv2[2:], self.bn2[2:], out2)
        
        if self.aggregation:
            x_shape = x.shape
            x = x.view(x_shape[0]//4, 4, x_shape[1], x_shape[2], x_shape[3]).contiguous()
            x = x.mean(dim=1)
            out2 = self.attention_layer(out2)
        
        out3 = self.residual_block(self.conv3[:2], self.bn3[:2], out2)
        out3 = self.residual_block(self.conv3[2:], self.bn3[2:], out3)
        
        out4 = self.conv4(out3)
        out4 = self.bn4(out4)
        out4 = self.relu(out4)
        
        out5 = self.conv5(out4)
        return x + out5


    
# Pixel-wise attention layer
class PixelWiseAttention(nn.Module):
    def __init__(self, num_channels, d_embed, dropout=0.1):
        super().__init__()
        self.linear_layer = nn.Linear(d_embed, 1)
        self.attention_value = nn.Linear(d_embed, d_embed)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.softmax_layer = nn.Softmax(dim=-1)
        self.scale = 1 / np.sqrt(d_embed)
        
        nn.init.xavier_uniform_(self.linear_layer.weight)
        nn.init.zeros_(self.linear_layer.bias)
        nn.init.xavier_uniform_(self.attention_value.weight)
        nn.init.zeros_(self.attention_value.bias)
        
    def forward(self, x):
        """
        x : (4*n_batch, channel, height, width)
        """
        x_shape = x.shape
        out = x.view(x_shape[0]//4, 4, x_shape[1], x_shape[2], x_shape[3]).contiguous().permute(0, 3, 4, 1, 2)
        
        scores = self.linear_layer(out).squeeze(dim=-1) * self.scale
        weights = self.softmax_layer(scores)
        out = torch.matmul(weights.unsqueeze(-2), self.attention_value(out)).squeeze(-2)
        return self.dropout_layer(out.permute(0, 3, 1, 2).contiguous())