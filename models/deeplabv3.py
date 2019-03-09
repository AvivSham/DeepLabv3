import torch
import torch.nn as nn
import torch.nn.functional as F

from .assp import ASSP
from .resnet_50 import ResNet_50

class DeepLabv3(nn.Module):
  
  def __init__(self, nc):
    
    super(DeepLabv3, self).__init__()
    
    self.nc = nc
    
    self.resnet = ResNet_50()
    
    self.assp = ASSP(in_channels = 1024)
    
    self.conv = nn.Conv2d(in_channels = 256, out_channels = self.nc,
                          kernel_size = 1, stride=1, padding=0)
        
  def forward(self,x):
    _, _, h, w = x.shape
    x = self.resnet(x)
    x = self.assp(x)
    x = self.conv(x)
    x = F.interpolate(x, size=(h, w), mode='bilinear') #scale_factor = 16, mode='bilinear')
    return x
