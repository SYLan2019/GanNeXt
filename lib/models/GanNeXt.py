""" Network architectures.
"""
# pylint: disable=W0221,W0622,C0103,R0913

##
import torch
import torch.nn as nn
import torch.nn.parallel
import functools
from torch.optim import lr_scheduler
from torch.nn import init
import numpy as np
from timm.models.layers import trunc_normal_, DropPath
import torch.nn.functional as F



class UnetGenerator(nn.Module):
    def __init__(self, input_nc=3,dim=40,size=32,norm='bn'):
        super(UnetGenerator, self).__init__()
        self.dim=dim
        deep_down=[2,2,6]
        deep_bottom=4
        deep_up=[2,2,6]
        k=2
        s=1
        size=size//k
        this_norm=nn.BatchNorm2d(dim) if norm=='bn' else LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.depth = len(deep_down)
        self.down = nn.ModuleList()
        self.skip = nn.ModuleList()
        self.up = nn.ModuleList()
        self.stem_in= nn.Sequential(
            nn.Conv2d(input_nc, dim, kernel_size=k, stride=k),
            this_norm
        )
        self.stem_out= nn.Sequential(nn.ConvTranspose2d( dim,dim*2, kernel_size=k, stride=k), nn.Conv2d(dim*2, input_nc, kernel_size=1, stride=1))
        for i in range(self.depth):         
            self.skip.append(nn.Sequential(*[Block(self.dim * (2 ** (i)),norm=norm) for _ in range(s)]))
            self.down.append(DownBlock(self.dim * (2 ** i),self.dim * (2 ** (i+1)),deep=deep_down[i],size=size//(2 ** i),norm=norm))
            j=self.depth - i - 1
            self.up.append(UpBlock(self.dim * (2 ** (j+1)),self.dim * (2 ** j),deep=deep_up[j],norm=norm))

        self.bottom=nn.Sequential(*[Block(self.dim * (2 ** self.depth),norm=norm)for _ in range(deep_bottom)])

    def forward(self, x):
        x=self.stem_in(x)
        skip=[]
        for i in range(self.depth):
            s,x=self.down[i](x)
            #skip.append(self.skip[i](s))
            skip.append(s)
        x=self.bottom(x)
        for i in range(self.depth):
            x=self.up[i](x,skip[self.depth-i-1])

        x=self.stem_out(x)
        return x
        
        
class Block(nn.Module):

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6,radio=4,norm='bn'):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim) # depthwise conv
        self.norm = nn.BatchNorm2d(dim) if norm=='bn' else LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.pwconv1 = nn.Linear(dim, radio * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),requires_grad=True) if layer_scale_init_value > 0 else None
        self.pwconv2 = nn.Linear(radio * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        
    
class DownBlock(nn.Module):
    def __init__(self, input_nc=3, output_nc=64, deep=2,size=8,norm='bn'):
        super(DownBlock, self).__init__()
       
        self.block = nn.Sequential(*[Block(input_nc,norm=norm) for _ in range(deep)])
        self.down= Down(input_nc,output_nc)
    
        
    def forward(self, x):

        x=self.block(x)        
        out=self.down(x)        
        return x ,out
    
class Down(nn.Module):

    def __init__(self, in_dim,out_dim, drop_path=0.,norm='bn'):
        super().__init__()
        #self.norm = LayerNorm(in_dim, eps=1e-6, data_format="channels_first")
        self.norm = nn.BatchNorm2d(out_dim) if norm=='bn' else LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.maxpool = nn.MaxPool2d(2,2)
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=2,stride=2) 
        self.conv2 = nn.Conv2d(in_dim+out_dim, out_dim, kernel_size=1,stride=1) 
       

    def forward(self, x):
        
        x_max=self.maxpool(x)
        x = self.conv1(x)
        x=torch.cat([x,x_max],dim=1)
        x=self.conv2(x)
        x=self.norm(x)
        return x

class Up(nn.Module):

    def __init__(self, in_dim,out_dim, drop_path=0.,kernel_size=2):
        super().__init__()
        self.dwconv = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=kernel_size,stride=kernel_size) # depthwise conv

    def forward(self, x):
        x = self.dwconv(x)
        return x 

          
class UpBlock(nn.Module):
    def __init__(self,  input_nc=64, output_nc=3,deep=2,norm='bn'):
        super(UpBlock, self).__init__()
        self.block = nn.Sequential(*[Block(output_nc,norm=norm) for _ in range(deep)])
        self.up=Up(input_nc,output_nc)
        self.proj=nn.Conv2d(output_nc*2,output_nc,kernel_size=1,stride=1)
            
    def forward(self, x,x1):
        x=self.up(x)
        x=torch.cat([x,x1],1)
        x=self.proj(x)
        x=self.block(x)
        
        return x
