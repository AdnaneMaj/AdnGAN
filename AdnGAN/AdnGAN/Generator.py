import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    def __init__(self,z_dim=100,channels_img=3,features_g=64,image_size=256,cond = False,num_classes=None):
        super(Generator,self).__init__()
        
        num_layers = int(np.log2(image_size/8))
        n = int(image_size/4)
        layers = [self._block(z_dim,features_g*n,4,1,0)]
        for _ in range(num_layers):
            layers.append(self._block(features_g*n,features_g*n//2,4,2,1))
            n = n//2
        layers.append(nn.ConvTranspose2d(features_g*2,channels_img,4,2,1))
        layers.append(nn.Tanh())
        self.conv = nn.Sequential(*layers)

        if cond:
            """
           Embedding for conditional GANs 
            """
            self.embed = nn.Embedding(
                num_classes,image_size*image_size
            )

    def _block(self,in_c,out_c,kernel_size,stride,padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c,out_c,kernel_size,stride,padding,bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2) #nn.ReLU()
        )

    def forward(self,z):
        return self.conv(z)