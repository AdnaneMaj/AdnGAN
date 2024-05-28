import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2

__all__ = ["Generator_pro","Discriminator_pro"]

#equalized leaning rate for a conv2d
class WSConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,padding=1,gain=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding)
        self.scale = (gain / (in_channels * (kernel_size ** 2))) ** 0.5
        self.bias = self.conv.bias #We are copying the bias of the current conv layer, we don't want the bias to be scaled, only the weights
        self.conv.bias = None #we remove the bias (i don't fucking understand why hhhhh)

        #initlaise conv layer
        nn.init.normal_(self.conv.weight) #initalise from normal disterbution
        nn.init.zeros_(self.bias) #initialise with zeros

    def forward(self,x):
        return self.conv(x*self.scale)+self.bias.view(1,self.bias.shape[0],1,1)

#Pixel norm class (instead of batch norm)
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
        #epsilon : 1e-8

    def forward(self,x):
        return x/torch.sqrt(torch.mean(x**2,dim=1,keepdim=True)+1e-8) #dim=1 : The mean accros the channels since dim=0 correspend to the batch

class ConvBlock(nn.Module):
    """
    This block will be used in the G and D
    We will use the conv2D using the equalized learning rate (initialisation)
    """
    def __init__(self,in_channels,out_channels,use_pixelnorm=True):
        super().__init__()
        self.conv1 = WSConv2d(in_channels,out_channels)
        self.conv2 = WSConv2d(out_channels,out_channels)
        self.leaky = nn.LeakyReLU(0.2)
        self.pn = PixelNorm()
        self.use_pn = use_pixelnorm

    def forward(self,x):
        x = self.leaky(self.conv1(x))
        x = self.pn(x) if self.use_pn else x
        x = self.leaky(self.conv2(x))
        x = self.pn(x) if self.use_pn else x
        return x
    

class Generator_pro(nn.Module):
    def __init__(self,z_dim,in_channels,img_channels=3,factors=[1,1,1,1,1/2,1/4,1/8,1/16,1/32]):
        super(Generator_pro,self).__init__()
        self.initial = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(z_dim,in_channels,kernel_size=4,stride=1,padding=0), #It take 1*1 -> 4*4
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm(),
        )

        self.initial_rgb = WSConv2d(in_channels,img_channels,kernel_size=1,stride=1,padding=0)
        self.prog_blocks,self.rgb_layers = nn.ModuleList([]),nn.ModuleList([self.initial_rgb]) #progressive blocs and rgb_layers

        for i in range(len(factors)-1):
            #factors[i] => factors[i+1]
            conv_in_c = int(in_channels*factors[i])
            conv_out_c = int(in_channels*factors[i+1])
            self.prog_blocks.append(ConvBlock(conv_in_c,conv_out_c))
            self.rgb_layers.append(WSConv2d(conv_out_c,img_channels,kernel_size=1,stride=1,padding=0))

    def fade_in(self,alpha,upscaled,generated):
        return torch.tanh(alpha*generated+(1-alpha)*upscaled) #[-1,1]

    def forward(self,x,alpha,steps): # step=0(4*4) steps=1 (8*8) ...
        out = self.initial(x) #4*4
        if steps == 0 :
            return self.initial_rgb(out)
        
        for step in range(steps):
            upscaled = F.interpolate(out,scale_factor=2,mode="nearest") #upscale
            out = self.prog_blocks[step](upscaled) #run throught the prog block

        final_upscaled = self.rgb_layers[steps-1](upscaled)
        final_out = self.rgb_layers[steps](out)
        return self.fade_in(alpha,final_upscaled,final_out)
    

class Discriminator_pro(nn.Module):
    def __init__(self,in_channels,img_channels=3,factors=[1,1,1,1,1/2,1/4,1/8,1/16,1/32]):
        super().__init__()
        self.prog_blocks,self.rgb_layers = nn.ModuleList([]),nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)

        for i in range(len(factors)-1,0,-1):
            conv_in_c = int(in_channels*factors[i])
            conv_out_c = int(in_channels*factors[i-1])
            self.prog_blocks.append(ConvBlock(conv_in_c,conv_out_c,use_pixelnorm=False))
            self.rgb_layers.append(WSConv2d(img_channels,conv_in_c,kernel_size=1,stride=1,padding=0))

        #This for 4*4 img resolution
        self.initial_rgb = WSConv2d(img_channels,in_channels,kernel_size=1,stride=1,padding=0)
        self.rgb_layers.append(self.initial_rgb)
        self.avg_pool = nn.AvgPool2d(kernel_size=2,stride=2)

        #block for 4*4 resolution
        self.final_block = nn.Sequential(
            WSConv2d(in_channels+1,in_channels,kernel_size=3,stride=1,padding=1), #513*4*4 to 512*4*4 (last block)
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels,in_channels,kernel_size=4,stride=1,padding=0),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels,1,kernel_size=1,stride=1,padding=0) #In the paper they did a linear layer (he said that it's the same thing)
        )


    def fade_in(self,alpha,downscaled,out):
        return alpha*out+(1-alpha)*downscaled

    def minibatch_std(self,x):
        batch_statistics = torch.std(x,dim=0).mean().repeat(x.shape[0],1,x.shape[2],x.shape[3]) #The std of every example of x N*C*H*W ==> N
        return torch.cat([x,batch_statistics],dim=1)

    def forward(self,x,alpha,steps): # steps = 0 (4*4), steps=1 (8*8) , etc6
        cur_step = len(self.prog_blocks)-steps
        out = self.leaky(self.rgb_layers[cur_step](x))

        if steps == 0:
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0],-1)
        
        downscaled = self.leaky(self.rgb_layers[cur_step+1](self.avg_pool(x)))
        out = self.avg_pool(self.prog_blocks[cur_step](out))
        out = self.fade_in(alpha,downscaled,out)

        for step in range(cur_step+1,len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)

        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0],-1)
    

if __name__ == "__main__":
    Z_DIM = 50
    IN_CHANNELS = 256
    gen = Generator_pro(Z_DIM,IN_CHANNELS)
    critic = Discriminator_pro(IN_CHANNELS)

    for img_size in [4,8,16,32,64,128,256,512,1024]:
        num_steps = int(log2(img_size/4))
        x = torch.randn(1,Z_DIM,1,1)
        z = gen(x,0.5,steps=num_steps)
        assert z.shape == (1,3,img_size,img_size)
        out = critic(z,alpha=0.5,steps=num_steps)
        assert out.shape == (1,1)
        print(f"Succes at image size :{img_size}")