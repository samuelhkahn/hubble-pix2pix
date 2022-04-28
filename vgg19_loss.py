from vgg19 import Vgg19
import torch.nn as nn
from torchvision import transforms 

class VGGLoss(nn.Module):
    def __init__(self,device,weights):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().to(device)
        self.criterion = nn.MSELoss()
        self.weights = weights      
    def forward(self, x, y):
        # x_norm,y_norm = self.normalize(x),self.normalize(y)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss+=self.weights[i]*self.criterion(x_vgg[i], y_vgg[i].detach())    
        return loss