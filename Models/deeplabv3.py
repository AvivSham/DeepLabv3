from .assp import ASSP
from .resnet-50 import ResNet_50


class DeepLabv3(nn.Module):
  def __init__(self, nc):
    super(DeepLabv3, self).__init__()
    
    self.nc = nc
    
    self.resnet = ResNet_50()
    
    self.assp = ASSP(nc = self.nc)
    
  def forward(self,x):
    
    x = self.resnet(x)
    x = self.assp(x)
    x = F.interpolate(x, size=(512, 512), mode='bilinear')
    
    return x
