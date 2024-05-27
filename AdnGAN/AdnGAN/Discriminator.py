import torch
import torch.nn as nn
import numpy as np

class Discriminator(nn.Module):
    def __init__(self,channels_img=3,features_d=64,image_size=256):
        super(Discriminator,self).__init__()

        num_layers = int(np.log2(image_size/8))
        layers = [nn.Conv2d(channels_img,features_d,4,2,1),nn.LeakyReLU(0.2)]
        n = 1
        for _ in range(num_layers):
            layers.append(self._block(features_d*n,features_d*n*2,4,2,1))
            n = n*2
        layers.append(nn.Conv2d(features_d*n,1,4,2,0))
        self.critic= nn.Sequential(*layers)

        """
        self.critic = nn.Sequential(
            nn.Conv2d(channels_img,features_d,4,2,1),
            nn.LeakyReLU(0.2),
            self._block(features_d,features_d*2,4,2,1),
            self._block(features_d*2,features_d*4,4,2,1),
            self._block(features_d*4,features_d*8,4,2,1),
            self._block(features_d*8,features_d*16,4,2,1),
            self._block(features_d*16,features_d*32,4,2,1),
            nn.Conv2d(features_d*32,1,4,2,0),
        )
        """

    def _block(self,in_c,out_c,kernel_size,stride,padding):
        return nn.Sequential(
            nn.Conv2d(in_c,out_c,kernel_size,stride,padding,bias=False),
            nn.InstanceNorm2d(out_c, affine=True),
            nn.LeakyReLU(0.2)
        )

    def forward(self,img):
        return self.critic(img)
    
def gradient_penalty(critic,real,fake,device):
    BATCH_SIZE,C,H,W = real.shape
    epsilon = torch.rand((BATCH_SIZE,1,1,1)).repeat(1,C,H,W).to(device)
    interpolated_images = real*epsilon+fake*(1-epsilon)

    #calcualte critic scores
    mixed_scores = critic(interpolated_images)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0] 

    gradient = gradient.view(gradient.shape[0],-1)
    gradient_norm = gradient.norm(2,dim=1)
    gradient_penalty = torch.mean((gradient_norm-1)**2)

    return gradient_penalty
