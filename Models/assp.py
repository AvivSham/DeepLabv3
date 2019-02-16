class ASSP(nn.Module):
  def __init__(self, nc,in_channels=1024, out_channels = 256):
    super(ASSP,self).__init__()
    
    self.nc = nc
    
    self.bn = nn.BatchNorm2d(out_channels)
    self.bnorm2 = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU()
    
    
    self.conv1 = nn.Conv2d(in_channels = in_channels, 
                          out_channels = out_channels,
                          kernel_size = 1,
                          padding = 0,
                          dilation=1)
    
    self.conv2 = nn.Conv2d(in_channels = in_channels, 
                          out_channels = out_channels,
                          kernel_size = 3,
                          padding = 6,
                          dilation = 6)
    
    self.conv3 = nn.Conv2d(in_channels = in_channels, 
                          out_channels = out_channels,
                          kernel_size = 3,
                          padding = 12,
                          dilation = 12)
    
    self.conv4 = nn.Conv2d(in_channels = in_channels, 
                          out_channels = out_channels,
                          kernel_size = 3,
                          padding = 18,
                          dilation = 18)
    
    self.conv5 = nn.Conv2d(in_channels = in_channels, 
                          out_channels = out_channels,
                          kernel_size = 1,
                          padding = 0,
                          dilation=1)
    
    self.convf = nn.Conv2d(in_channels = out_channels * 5, 
                          out_channels = nc,
                          kernel_size = 1,
                          padding = 0,
                          dilation=1)
    
    self.adapool = nn.AdaptiveAvgPool2d((16, 16))
   
  
  def forward(self,x):
    
    x1 = self.conv1(x)
    x1 = self.bn(x1)
    x1 = self.relu(x1)
    
    x2 = self.conv2(x)
    x2 = self.bn(x2)
    x2 = self.relu(x2)
    
    x3 = self.conv3(x)
    x3 = self.bn(x3)
    x3 = self.relu(x3)
    
    x4 = self.conv4(x)
    x4 = self.bn(x4)
    x4 = self.relu(x4)
    
    x5 = self.adapool(x)
    x5 = self.conv5(x5)
    x5 = self.bn(x5)
    x5 = self.relu(x5)
    
    print (x1.shape, x2.shape, x3.shape, x4.shape, x5.shape)
    x = torch.cat((x1,x2,x3,x4,x5), dim = 1) #channels first
    x = self.convf(x)
    x = self.bnorm2(x)
    x = self.relu(x)
    
    return x
