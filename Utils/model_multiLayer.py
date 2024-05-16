import numpy as np
import torch.nn as nn
from torchvision import datasets, models, transforms
import os, torchvision, torch
import torch.nn.functional as F
from collections import OrderedDict
from functools import partial

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




class AttentionSlide_MultiBatch(nn.Module):
    def __init__(self):
        super(AttentionSlide_MultiBatch, self).__init__()
        self.L = 1000
        self.D = 128
        self.K = 1
        
        self.resnet = models.resnet34(pretrained=True)
        

        self.attention1 = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier1 = nn.Linear(self.L*self.K, 1)

        self.attention2 = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier2 = nn.Linear(self.L*self.K, 1)

        self.flatten = nn.Flatten()

    def forward(self, x):
        x1 = x.view(x.shape[0]*x.shape[1], 3, x.shape[3], x.shape[4]) #squeeze batchsize for conv parallel.
        H = self.resnet(x1)
        A1 = self.attention1(H)  # NxK
        A2 = self.attention2(H)
        # print(A.shape)
        A1 = A1.view(x.shape[0], x.shape[1], -1)# recovery batchsize（
        A1 = torch.transpose(A1, 2, 1)  # KxN
        A2 = A2.view(x.shape[0], x.shape[1], -1)# recovery batchsize
        A2 = torch.transpose(A2, 2, 1)  # KxN
        # print(A.shape)
        A1 = F.softmax(A1, dim=2)  # softmax over N
        A2 = F.softmax(A2, dim=2)
        H = H.view(x.shape[0], x.shape[1], -1)
        M1 = torch.matmul(A1, H)  # KxL attention to channel
        M1 = M1.squeeze(dim=1)
        M1 = self.flatten(M1)
        M2 = torch.matmul(A2, H)  # KxL attention to channel
        M2 = M2.squeeze(dim=1)
        M2 = self.flatten(M2)
        # print(M.shape)
        wild_prob1 = self.classifier1(M1)
        wild_prob1 = wild_prob1.squeeze()
        wild_prob2 = self.classifier2(M2)
        wild_prob2 = wild_prob2.squeeze()
        Y_prob = torch.stack((wild_prob1,wild_prob2),dim=1)
        _, Y_hat = torch.max(Y_prob, 1)
        A = torch.stack((A1,A2),dim=1).squeeze()

        return Y_prob, Y_hat, A
    











    


        



class FeatureExtractor(nn.Module):
    def __init__(self,layer_name):
        super(FeatureExtractor, self).__init__()

        self.base_net = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        # self.base_net = ResNetLayerNorm()

        self.layer_name = layer_name
    def forward(self, x_40,x_10,x_down_10):
        ### x_40:[batch,bag(16),3,512,512]
        ### x_10:[batch,1,3,512,512]
        
        x_40_view = x_40.view(x_40.shape[0]*x_40.shape[1],3,x_40.shape[3],x_40.shape[4])
        x_10 = x_10.squeeze(dim = 1)
        x_down_10 = x_down_10.squeeze(dim = 1)

        # print(x_10.size())
        x_10_view = x_10.view(x_10.shape[0],3,x_10.shape[2],x_10.shape[3])
        x_down_10_view = x_down_10.view(x_down_10.shape[0],3,x_down_10.shape[2],x_down_10.shape[3])


        for name,module in self.base_net._modules.items():
            # print(name)
            # print(module)
            x_40_view=module(x_40_view)
            # print(x_40_view.size())
            x_10_view=module(x_10_view)
            x_down_10_view=module(x_down_10_view)


            if name == self.layer_name:
                return x_40_view,x_10_view,x_down_10_view




class Fea_Layer1(nn.Module):
    def __init__(self):
        super(Fea_Layer1, self).__init__()
        
        self.FeaExtractor = FeatureExtractor(layer_name='layer1')
        self.conv = nn.Conv3d(16, 1, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv_attention = nn.Conv3d(in_channels=64, out_channels=1, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))

    def forward(self, x_40,x_10,x_down_10):
        ####(batch*bag,64,128,128) 
        ###(batch,64,128,128)
        ###(batch,64,32,32)
        fea_layer1_40,fea_layer1_10 ,fea_layer1_down10 =  self.FeaExtractor(x_40,x_10,x_down_10)
 
        ###处理40倍的数据，bag（16）到 1
        ##(batch,bag,64,128,128)
        fea_layer1_40_view = fea_layer1_40.view(x_40.shape[0],x_40.shape[1],fea_layer1_40.shape[1],fea_layer1_40.shape[2],fea_layer1_40.shape[3])
        ##(batch,1,64,128,128)
        f_layer1_40 = self.conv(fea_layer1_40_view)
        ##(batch,64,128,128)
        f_layer1_40 = f_layer1_40.squeeze(dim = 1)
        

        return f_layer1_40,fea_layer1_10,fea_layer1_down10


class Fea_Layer2(nn.Module):
    def __init__(self):
        super(Fea_Layer2, self).__init__()
        
        self.FeaExtractor = FeatureExtractor(layer_name='layer2')
       

        self.conv = nn.Conv3d(16, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_attention = nn.Conv3d(in_channels=128, out_channels=1, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))

    def forward(self,  x_40,x_10,x_down_10):
        ####(batch*bag,128,64,64)
        ###(batch,128,64,64)
        ###(batch,128,16,16)

        fea_layer2_40,fea_layer2_10,fea_layer2_down10 = self.FeaExtractor( x_40,x_10,x_down_10)
       

        ##(batch,bag,128,64,64)
        fea_layer2_40_view = fea_layer2_40.view(x_40.shape[0],x_40.shape[1],fea_layer2_40.shape[1],fea_layer2_40.shape[2],fea_layer2_40.shape[3])
        ##(batch,1,128,64,64)
        f_layer2_40 = self.conv(fea_layer2_40_view)
        ##(batch,128,64,64)
        f_layer2_40 = f_layer2_40.squeeze(dim = 1)
   

        return f_layer2_40,fea_layer2_10,fea_layer2_down10




class Fea_Layer3(nn.Module):
    def __init__(self):
        super(Fea_Layer3, self).__init__()
        
        self.FeaExtractor = FeatureExtractor(layer_name='layer3')


        self.conv = nn.Conv3d(16, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_attention = nn.Conv3d(in_channels=256, out_channels=1, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))

    def forward(self,x_40,x_10,x_down_10):
        ####(batch*bag,256,32,32)
        ###(batch,256,32,32)
        ###(batch,256,8,8)
        fea_layer3_40,fea_layer3_10,fea_layer3_down10 = self.FeaExtractor(x_40,x_10,x_down_10)
       
        ##(batch,bag,256,32,32)
        fea_layer3_40_view = fea_layer3_40.view(x_40.shape[0],x_40.shape[1],fea_layer3_40.shape[1],fea_layer3_40.shape[2],fea_layer3_40.shape[3])
        ##(batch,1,256,32,32)
        f_layer3_40 = self.conv(fea_layer3_40_view)
      
        ##(batch,256,32,32)
        f_layer3_40 = f_layer3_40.squeeze(dim = 1)
     
        return f_layer3_40,fea_layer3_10,fea_layer3_down10




class Fea_Layer4(nn.Module):
    def __init__(self):
        super(Fea_Layer4, self).__init__()
        
        self.FeaExtractor = FeatureExtractor(layer_name='layer4')

        self.conv = nn.Conv3d(16, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_attention = nn.Conv3d(in_channels=512, out_channels=1, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))

    def forward(self, x_40,x_10,x_down_10):
        ####(batch*bag,512,16,16)
        ###(batch,512,16,16)
        ###(batch,512,4,4)
        fea_layer4_40,fea_layer4_10,fea_layer4_down10 = self.FeaExtractor(x_40,x_10,x_down_10)
     
        ##(batch,bag,512,16,16)
        fea_layer4_40_view = fea_layer4_40.view(x_40.shape[0],x_40.shape[1],fea_layer4_40.shape[1],fea_layer4_40.shape[2],fea_layer4_40.shape[3])

        ##(batch,1,512,16,16)
        f_layer4_40 = self.conv(fea_layer4_40_view)
        ##(batch,512,16,16)
        f_layer4_40 = f_layer4_40.squeeze(dim = 1)
      

        return f_layer4_40,fea_layer4_10,fea_layer4_down10








class keyBlock_new(nn.Module):
    def __init__(self):
        super(keyBlock_new, self).__init__()

        ##定义参数a、b、c、d作为模型的可训练参数，并使用随机初始化
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))
        self.c = nn.Parameter(torch.randn(1))
        self.d = nn.Parameter(torch.randn(1))

        
        self.fea_layer1 = Fea_Layer1()
        self.fea_layer2 = Fea_Layer2()
        self.fea_layer3 = Fea_Layer3()
        self.fea_layer4 = Fea_Layer4()

        self.inplanes_layer1 = 64  ##C通道数
        self.inplanes_layer2 = 128  ##C通道数
        self.inplanes_layer3 = 256  ##C通道数
        self.inplanes_layer4 = 512  ##C通道数

        
        self.conv1 = nn.Conv2d( in_channels = self.inplanes_layer1,out_channels = self.inplanes_layer1,kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d( in_channels = self.inplanes_layer2,out_channels = self.inplanes_layer2,kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d( in_channels = self.inplanes_layer3,out_channels = self.inplanes_layer3,kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d( in_channels = self.inplanes_layer4,out_channels = self.inplanes_layer4,kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(self.inplanes_layer1 )
        self.bn2 = nn.BatchNorm2d(self.inplanes_layer2 )
        self.bn3 = nn.BatchNorm2d(self.inplanes_layer3 )
        self.bn4 = nn.BatchNorm2d(self.inplanes_layer4 )


        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()

        # self.dense_layer1 = nn.Sequential(
        #     F.interpolate(input, size=64,mode='bilinear'),
        #     # nn.ConvTranspose2d(in_channels=self.inplanes_layer1, out_channels=self.inplanes_layer1, kernel_size=4, stride=2, padding=1),
            
        # )
    
        # self.dense_layer2 = nn.Sequential(
        #     F.interpolate(input, size=32,mode='bilinear'),

        #     # nn.ConvTranspose2d(in_channels=self.inplanes_layer2, out_channels=self.inplanes_layer2, kernel_size=4, stride=2, padding=1),

        # )
        # self.dense_layer3 = nn.Sequential(
        #     F.interpolate(input, size=16,mode='bilinear'),

        #     # nn.ConvTranspose2d(in_channels=self.inplanes_layer3, out_channels=self.inplanes_layer3, kernel_size=4, stride=2, padding=1),

        # )
        # self.dense_layer4 = nn.Sequential(
        #     F.interpolate(input, size=8,mode='bilinear'),

        #     # nn.ConvTranspose2d(in_channels=self.inplanes_layer4, out_channels=self.inplanes_layer4, kernel_size=4, stride=2, padding=1),

        # )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
     

        self.fc_layer1 = nn.Sequential(
            nn.Linear(self.inplanes_layer1,8),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(8,2)
        )
        self.fc_layer2 = nn.Sequential(
            nn.Linear(self.inplanes_layer2,16),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(16,2)
        )
        self.fc_layer3 = nn.Sequential(
            nn.Linear(self.inplanes_layer3,32),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(32,2)
        )
        self.fc_layer4 = nn.Sequential(
            nn.Linear(self.inplanes_layer4,64),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(64,2)
        )

       

    def forward(self,x_40,x_10,x_down_10):


        # 使用softmax函数对a、b、c、d进行归一化，使它们之和为1
        parameters = torch.cat([self.a, self.b, self.c, self.d])
        normalized_parameters = torch.softmax(parameters, dim=0)
        
        ##f_layer1_40:(batch,64,64,64)
        ###fea_layer1_10:(batch,64,64,64)
        ####f_layer1_down10:(batch,64,32,32)
        f_layer1_40,fea_layer1_10 ,f_layer1_down10= self.fea_layer1(x_40,x_10,x_down_10)
        ##f_layer2_40:(batch,128,32,32)
        ###fea_layer2_10:(batch,128,32,32)
        ####f_layer2_down10:(batch,128,16,16)
        f_layer2_40,fea_layer2_10,f_layer2_down10 = self.fea_layer2(x_40,x_10,x_down_10)
        ##f_layer3_40:(batch,256,16,16)
        ###fea_layer3_10:(batch,256,16,16)
        ####f_layer3_down10:(batch,256,8,8)
        f_layer3_40,fea_layer3_10,f_layer3_down10 = self.fea_layer3(x_40,x_10,x_down_10)
        ##f_layer4_40:(batch,512,8,8)
        ###fea_layer4_10:(batch,512,8,8)
        ####f_layer4_down10:(batch,512,4,4)
        f_layer4_40,fea_layer4_10,f_layer4_down10= self.fea_layer4(x_40,x_10,x_down_10)

   
        ###10倍降采样4倍的128*128的图块进行转置卷积(上采样）变成与10同样的tensor大小
        
        f_layer1_d10_inter =  F.interpolate(f_layer1_down10, size=64,mode='nearest')  
        f_layer2_d10_inter = F.interpolate(f_layer2_down10, size=32,mode='nearest')  
        f_layer3_d10_inter = F.interpolate(f_layer3_down10, size=16,mode='nearest')  
        f_layer4_d10_inter = F.interpolate(f_layer4_down10, size=8,mode='nearest')  


        #######f_layer1_d10_inter:(batch,64,64,64)
        ##反卷积后进行3*3的维的卷积
        f_layer1_d10_inter = self.tanh(self.bn1(self.conv1(f_layer1_d10_inter)))##维度不变
        f_layer2_d10_inter = self.tanh(self.bn2(self.conv2(f_layer2_d10_inter)))##维度不变
        f_layer3_d10_inter = self.tanh(self.bn3(self.conv3(f_layer3_d10_inter)))##维度不变
        f_layer4_d10_inter = self.tanh(self.bn4(self.conv4(f_layer4_d10_inter)))##维度不变

        ##f_layer1_10:(batch,64,64,64)
        ##10倍的也进行3的卷积
        f_layer1_10_ = self.tanh(self.bn1(self.conv1(fea_layer1_10)))##维度不变
        f_layer2_10_ = self.tanh(self.bn2(self.conv2(fea_layer2_10)))##维度不变
        f_layer3_10_ = self.tanh(self.bn3(self.conv3(fea_layer3_10)))##维度不变
        f_layer4_10_ = self.tanh(self.bn4(self.conv4(fea_layer4_10)))##维度不变



        ##点乘##(batch,64,64,64)
        ##10倍和10倍降采样的进行点乘
        f_Lay1_10_mul = torch.mul(f_layer1_d10_inter,f_layer1_10_)
        f_Lay2_10_mul = torch.mul(f_layer2_d10_inter,f_layer2_10_)
        f_Lay3_10_mul = torch.mul(f_layer3_d10_inter,f_layer3_10_)
        f_Lay4_10_mul = torch.mul(f_layer4_d10_inter,f_layer4_10_)


        ##点乘后在进行3的卷积
        f_Lay1_10_mul = self.tanh(self.bn1(self.conv1(f_Lay1_10_mul)))##维度不变
        f_Lay2_10_mul = self.tanh(self.bn2(self.conv2(f_Lay2_10_mul)))##维度不变
        f_Lay3_10_mul = self.tanh(self.bn3(self.conv3(f_Lay3_10_mul)))##维度不变
        f_Lay4_10_mul = self.tanh(self.bn4(self.conv4(f_Lay4_10_mul)))##维度不变

        ##将点乘后（融合10倍和10倍降采样四倍的）在加上原来的10倍的特征
        f_Lay1_10_mul_add = f_layer1_10_ + f_Lay1_10_mul
        f_Lay2_10_mul_add = f_layer2_10_ + f_Lay2_10_mul
        f_Lay3_10_mul_add = f_layer3_10_ + f_Lay3_10_mul
        f_Lay4_10_mul_add = f_layer4_10_ + f_Lay4_10_mul


        ####### 融合后进行3*3的卷积
        f_Lay1_10_mul_add = self.tanh(self.bn1(self.conv1(f_Lay1_10_mul_add)))##维度不变
        f_Lay2_10_mul_add = self.tanh(self.bn2(self.conv2(f_Lay2_10_mul_add)))##维度不变
        f_Lay3_10_mul_add = self.tanh(self.bn3(self.conv3(f_Lay3_10_mul_add)))##维度不变
        f_Lay4_10_mul_add = self.tanh(self.bn4(self.conv4(f_Lay4_10_mul_add)))##维度不变

        
        ##处理40倍的，只进行3的卷积
        fea_layer1_40 = self.tanh(self.bn1(self.conv1(f_layer1_40)))##维度不变
        fea_layer2_40 = self.tanh(self.bn2(self.conv2(f_layer2_40)))##维度不变
        fea_layer3_40 = self.tanh(self.bn3(self.conv3(f_layer3_40)))##维度不变
        fea_layer4_40 = self.tanh(self.bn4(self.conv4(f_layer4_40)))##维度不变
        


        ###处理好的10x的加上40倍数的
        ###batch*64*128*128
        fea_merge_layer1 = f_Lay1_10_mul_add + fea_layer1_40
        fea_merge_layer2 = f_Lay2_10_mul_add + fea_layer2_40
        fea_merge_layer3 = f_Lay3_10_mul_add + fea_layer3_40
        fea_merge_layer4 = f_Lay4_10_mul_add + fea_layer4_40

        ###相加后在进行3的卷积
        ###batch*64*64*64
        fea_merge_layer1 = self.tanh(self.bn1(self.conv1(fea_merge_layer1)))##维度不变
        fea_merge_layer2 = self.tanh(self.bn2(self.conv2(fea_merge_layer2)))##维度不变
        fea_merge_layer3 = self.tanh(self.bn3(self.conv3(fea_merge_layer3)))##维度不变
        fea_merge_layer4 = self.tanh(self.bn4(self.conv4(fea_merge_layer4)))##维度不变





        ###将融合后的特征接transformer进行更好的融合








        ##batch*64*1*1
        fea_merge_layer1 = self.avgpool(fea_merge_layer1)
        fea_merge_layer2 = self.avgpool(fea_merge_layer2)
        # print(fea_merge_layer2.size())
        fea_merge_layer3 = self.avgpool(fea_merge_layer3)
        fea_merge_layer4 = self.avgpool(fea_merge_layer4)

        #batch*64
        fea_merge_layer1 = torch.flatten(fea_merge_layer1, 1)
        fea_merge_layer2 = torch.flatten(fea_merge_layer2, 1)
        # print(fea_merge_layer2.size())
        fea_merge_layer3 = torch.flatten(fea_merge_layer3, 1)
        fea_merge_layer4 = torch.flatten(fea_merge_layer4, 1)

        # # fea_all = torch.cat((fea_merge_layer1, fea_merge_layer2,fea_merge_layer3,fea_merge_layer4), dim=1)

        # #batch*2
        out_layer1 = self.fc_layer1(fea_merge_layer1)
        out_layer2 = self.fc_layer2(fea_merge_layer2)
        # print(out_layer2.size())
        out_layer3 = self.fc_layer3(fea_merge_layer3)
        out_layer4 = self.fc_layer4(fea_merge_layer4)
        # # out_layer = self.fc_layer(fea_all)

        # ##batch*2
        Y_layer1_prob=F.softmax(out_layer1,dim=1)
        Y_layer2_prob=F.softmax(out_layer2,dim=1)
        Y_layer3_prob=F.softmax(out_layer3,dim=1)
        Y_layer4_prob=F.softmax(out_layer4,dim=1)
        # # Y_prob=F.softmax(out_layer,dim=1)


        # # 计算a*loss1 + b*loss2 + c*loss3 + d*loss4
        Y_prob = normalized_parameters[0]*Y_layer1_prob + normalized_parameters[1]*Y_layer2_prob + normalized_parameters[2]*Y_layer3_prob + normalized_parameters[3]*Y_layer4_prob


        # # Y_prob = Y_layer1_prob +  Y_layer2_prob + Y_layer3_prob + Y_layer4_prob
        # # Y_prob = self.param.a.item() * Y_layer1_prob + self.param.b.item() * Y_layer2_prob + self.param.c.item() * Y_layer3_prob + self.param.d.item() * Y_layer4_prob

        # # Y_prob = 0.1*out_layer1 + 0.2*out_layer2 + 0.3*out_layer3 + 0.4*out_layer4


        # # _40, Y_40_hat = torch.max(Y_40_prob, 1)
        # # _10, Y_10_hat = torch.max(Y_10_prob, 1)
        _, Y_hat = torch.max(Y_prob, 1)

        return Y_prob,Y_hat,Y_layer1_prob,Y_layer2_prob,Y_layer3_prob,Y_layer4_prob
        # return fea_merge_layer1,fea_merge_layer2,fea_merge_layer3,fea_merge_layer4
    






    ###muti head self-attention
##一层一个融合后的特征
###batch*64*64*64
class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=2,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 ):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        ##每个头的dim
        head_dim = dim // num_heads
        
        ##norm处理
        self.scale = qk_scale or head_dim ** -0.5

        ##qkv通过全连接生成
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        ##Wo的映射
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x_Layer):
        
       
        # [batch_size, num_patches + 1, total_embed_dim]
        B_layer, N_layer, C_layer = x_Layer.shape
      


        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x_Layer).reshape(B_layer, N_layer, 3, self.num_heads, C_layer // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @:矩阵乘法 multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        ##每个头都进行了矩阵乘法
        attn = (q @ k.transpose(-2, -1) ) * self.scale
        ##对每一行进行softmax
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        ##通过reshape实现head上的concat
        x = (attn @ v).transpose(1, 2).reshape(B_layer, N_layer, C_layer)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)



class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        ##[B, 197, 768]
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x






class my_VisionTransformer(nn.Module):

    ##depth：在transformer encoder中重复堆叠的L次数
    ##representation_size：在最后的mlp中的全连接层
    def __init__(self, num_classes=2,depth=2, num_heads=2, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0.,  norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(my_VisionTransformer, self).__init__()



        self.layer_Featurehook = keyBlock_new()

        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))
        self.c = nn.Parameter(torch.randn(1))
        self.d = nn.Parameter(torch.randn(1))


        self.num_classes = num_classes
        self.embed_dim_layer1 = 64  ##C通道数
        self.embed_dim_layer2 = 128  ##C通道数
        self.embed_dim_layer3 = 256  ##C通道数
        self.embed_dim_layer4 = 512  ##C通道数
        ##特征数量即C通道上的值
        # self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.num_tokens = 1

        ##H*W
        self.num_patches_layer1 = 4096
        self.num_patches_layer2 = 1024
        self.num_patches_layer3 = 256
        self.num_patches_layer4 = 64


        self.cls_token_layer1 = nn.Parameter(torch.zeros(1, 1, self.embed_dim_layer1))
        self.cls_token_layer2 = nn.Parameter(torch.zeros(1, 1, self.embed_dim_layer2))
        self.cls_token_layer3 = nn.Parameter(torch.zeros(1, 1, self.embed_dim_layer3))
        self.cls_token_layer4 = nn.Parameter(torch.zeros(1, 1, self.embed_dim_layer4))

        ##不用管，vit用不到，默认为none
        ##nn.Parameter 将一个固定不可训练的tensor转换成可以训练的类型parameter
        
        self.pos_embed_layer1 = nn.Parameter(torch.zeros(1, self.num_patches_layer1 + self.num_tokens, self.embed_dim_layer1))
        self.pos_embed_layer2 = nn.Parameter(torch.zeros(1, self.num_patches_layer2 + self.num_tokens, self.embed_dim_layer2))
        self.pos_embed_layer3 = nn.Parameter(torch.zeros(1, self.num_patches_layer3 + self.num_tokens, self.embed_dim_layer3))
        self.pos_embed_layer4 = nn.Parameter(torch.zeros(1, self.num_patches_layer4 + self.num_tokens, self.embed_dim_layer4))


        self.pos_drop = nn.Dropout(p=drop_ratio)
        ##构建等差序列0-depth，在l个block中drop_path_ratio是递增的
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        
        ##transformer的l个堆叠的blocks，是列表
        self.blocks_layer1 = nn.Sequential(*[
            Block(dim=self.embed_dim_layer1, num_heads=8, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])

        self.blocks_layer2 = nn.Sequential(*[
            Block(dim=self.embed_dim_layer2, num_heads=4, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])

        self.blocks_layer3 = nn.Sequential(*[
            Block(dim=self.embed_dim_layer3, num_heads=2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])

        self.blocks_layer4 = nn.Sequential(*[
            Block(dim=self.embed_dim_layer4, num_heads=1, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])



       

        self.norm_layer1 = norm_layer(self.embed_dim_layer1)
        self.norm_layer2 = norm_layer(self.embed_dim_layer2)
        self.norm_layer3 = norm_layer(self.embed_dim_layer3)
        self.norm_layer4 = norm_layer(self.embed_dim_layer4)


        
        self.has_logits = False
        self.pre_logits = nn.Identity()


        self.pre_logits_layer1 = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(self.embed_dim_layer1, 4)),
                ("act", nn.Tanh()),
                ("drop",nn.Dropout())
            ]))
        self.pre_logits_layer2 = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(self.embed_dim_layer2, 8)),
                ("act", nn.Tanh()),
                ("drop",nn.Dropout())

            ]))
        self.pre_logits_layer3 = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(self.embed_dim_layer3, 16)),
                ("act", nn.Tanh()),
                ("drop",nn.Dropout())

            ]))
        self.pre_logits_layer4 = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(self.embed_dim_layer4, 32)),
                ("act", nn.Tanh()),
                ("drop",nn.Dropout())

            ]))

        self.dist_token =  None
        # Classifier head(s)
        self.head_layer1 = nn.Linear(4 , num_classes) if num_classes > 0 else nn.Identity()
        self.head_layer2 = nn.Linear(8 , num_classes) if num_classes > 0 else nn.Identity()
        self.head_layer3 = nn.Linear(16 , num_classes) if num_classes > 0 else nn.Identity()
        self.head_layer4 = nn.Linear(32 , num_classes) if num_classes > 0 else nn.Identity()

        # self.head_dist = None
      
        # Weight init
        nn.init.trunc_normal_(self.pos_embed_layer1, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_layer2, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_layer3, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_layer4, std=0.02)

        
        nn.init.trunc_normal_(self.cls_token_layer1, std=0.02)
        nn.init.trunc_normal_(self.cls_token_layer2, std=0.02)
        nn.init.trunc_normal_(self.cls_token_layer3, std=0.02)
        nn.init.trunc_normal_(self.cls_token_layer4, std=0.02)

        # self.apply(_init_vit_weights)

    def forward_features(self, x_40,x_10,x_down_10):
        # [B, C, H, W] -> [B, num_patches, embed_dim]


        fea_layer1,fea_layer2,fea_layer3,fea_layer4 = self.layer_Featurehook(x_40,x_10,x_down_10)  # [B, 196, 768]


        fea_layer1_view = fea_layer1.view(fea_layer1.shape[0], fea_layer1.shape[2]*fea_layer1.shape[3], fea_layer1.shape[1] )
        fea_layer2_view = fea_layer2.view(fea_layer2.shape[0], fea_layer2.shape[2]*fea_layer2.shape[3], fea_layer2.shape[1] )
        fea_layer3_view = fea_layer3.view(fea_layer3.shape[0], fea_layer3.shape[2]*fea_layer3.shape[3], fea_layer3.shape[1] )
        fea_layer4_view = fea_layer4.view(fea_layer4.shape[0], fea_layer4.shape[2]*fea_layer4.shape[3], fea_layer4.shape[1] )




        # [1, 1, 768] -> [B, 1, 768]
        class_token_layer1 = self.cls_token_layer1.expand(fea_layer1_view.shape[0], -1, -1)
        class_token_layer2 = self.cls_token_layer2.expand(fea_layer2_view.shape[0], -1, -1)
        class_token_layer3 = self.cls_token_layer3.expand(fea_layer3_view.shape[0], -1, -1)
        class_token_layer4 = self.cls_token_layer4.expand(fea_layer4_view.shape[0], -1, -1)



       
        x_layer1 = torch.cat((class_token_layer1, fea_layer1_view), dim=1)  # [B, 197, 768]
        x_layer2 = torch.cat((class_token_layer2, fea_layer2_view), dim=1)  # [B, 197, 768]
        x_layer3 = torch.cat((class_token_layer3, fea_layer3_view), dim=1)  # [B, 197, 768]
        x_layer4 = torch.cat((class_token_layer4, fea_layer4_view), dim=1)  # [B, 197, 768]

       
        x_layer1 = self.pos_drop(x_layer1 + self.pos_embed_layer1)
        x_layer2 = self.pos_drop(x_layer2 + self.pos_embed_layer2)
        x_layer3 = self.pos_drop(x_layer3 + self.pos_embed_layer3)
        x_layer4 = self.pos_drop(x_layer4 + self.pos_embed_layer4)


        x_layer1 = self.blocks_layer1(x_layer1)
        x_layer2 = self.blocks_layer2(x_layer2)
        x_layer3 = self.blocks_layer3(x_layer3)
        x_layer4 = self.blocks_layer4(x_layer4)


        x_layer1 = self.norm_layer1(x_layer1) 
        x_layer2 = self.norm_layer2(x_layer2) 
        x_layer3 = self.norm_layer3(x_layer3) 
        x_layer4 = self.norm_layer4(x_layer4) 
        if self.dist_token is None:
            return self.pre_logits_layer1(x_layer1[:, 0]), self.pre_logits_layer2(x_layer2[:, 0]), self.pre_logits_layer3(x_layer3[:, 0]), self.pre_logits_layer4(x_layer4[:, 0])

        
        

    def forward(self, x_40,x_10,x_down_10):
        parameters = torch.cat([self.a, self.b, self.c, self.d])
        normalized_parameters = torch.softmax(parameters, dim=0)
        
        out_layer1,out_layer2,out_layer3,out_layer4 = self.forward_features(x_40,x_10,x_down_10)
        
        out_layer1 = self.head_layer1(out_layer1)
        out_layer2 = self.head_layer2(out_layer2)
        out_layer3 = self.head_layer3(out_layer3)
        out_layer4 = self.head_layer4(out_layer4)
        
         # ##batch*2
        Y_layer1_prob=F.softmax(out_layer1,dim=1)
        Y_layer2_prob=F.softmax(out_layer2,dim=1)
        Y_layer3_prob=F.softmax(out_layer3,dim=1)
        Y_layer4_prob=F.softmax(out_layer4,dim=1)
        # Y_prob=F.softmax(out_layer,dim=1)


        # 计算a*loss1 + b*loss2 + c*loss3 + d*loss4
        Y_prob = normalized_parameters[0]*Y_layer1_prob + normalized_parameters[1]*Y_layer2_prob + normalized_parameters[2]*Y_layer3_prob + normalized_parameters[3]*Y_layer4_prob


        # Y_prob = Y_layer1_prob +  Y_layer2_prob + Y_layer3_prob + Y_layer4_prob
        # Y_prob = self.param.a.item() * Y_layer1_prob + self.param.b.item() * Y_layer2_prob + self.param.c.item() * Y_layer3_prob + self.param.d.item() * Y_layer4_prob

        # Y_prob = 0.1*out_layer1 + 0.2*out_layer2 + 0.3*out_layer3 + 0.4*out_layer4


        # _40, Y_40_hat = torch.max(Y_40_prob, 1)
        # _10, Y_10_hat = torch.max(Y_10_prob, 1)
        _, Y_hat = torch.max(Y_prob, 1)

        return Y_prob,Y_hat
    


# def _init_vit_weights(m):
#     """
#     ViT weight initialization
#     :param m: module
#     """
#     weights_dict = torch.load(weights, map_location=device)

#     # if isinstance(m, nn.Linear):
#     #     nn.init.trunc_normal_(m.weight, std=.01)
#     #     if m.bias is not None:
#     #         nn.init.zeros_(m.bias)
#     # elif isinstance(m, nn.Conv2d):
#     #     nn.init.kaiming_normal_(m.weight, mode="fan_out")
#     #     if m.bias is not None:
#     #         nn.init.zeros_(m.bias)
#     # elif isinstance(m, nn.LayerNorm):
#     #     nn.init.zeros_(m.bias)
#     #     nn.init.ones_(m.weight)


 
#     if m == 'blocks_layer1' or m == 'blocks_layer2' or m == 'blocks_layer3' or m == 'blocks_layer4':
#         m.load_state_dict(weights_dict['block'])


