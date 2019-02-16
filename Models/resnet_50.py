class ResNet_50 (nn.Module):
  def __init__(self, in_channels = 3, conv1_out = 64):
    super(ResNet_50,self).__init__()
    
    self.in_channels = in_channels
    
    self.conv1_out = conv1_out
    
    self.resnet_50 = models.resnet50(pretrained = True)
    
    self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = conv1_out, 
                           kernel_size=(7,7), stride = 2)
    
    self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2)
  
  def forward(self,x):
    x = self.conv1(x)
    x = self.maxpool(x)
    x = self.resnet_50.layer1(x)
    x = self.resnet_50.layer2(x)
    x = self.resnet_50.layer3(x)
    
    
    return x
