import numpy as np
import torch.nn as nn
from torchvision import datasets, models, transforms
import os, torchvision, torch
import torch.nn.functional as F
from collections import OrderedDict
from functools import partial

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FeatureExtractor(nn.Module):
    def __init__(self,layer_name):
        super(FeatureExtractor, self).__init__()

        self.base_net = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        # self.base_net = ResNetLayerNorm()

        self.layer_name = layer_name
    def forward(self, x_10,x_down_10):
        ### x_40:[batch,bag(16),3,512,512]
        ### x_10:[batch,1,3,512,512]
        
        
        x_10 = x_10.squeeze(dim = 1)
        x_down_10 = x_down_10.squeeze(dim = 1)

        # print(x_10.size())
        x_10_view = x_10.view(x_10.shape[0],3,x_10.shape[2],x_10.shape[3])
        x_down_10_view = x_down_10.view(x_down_10.shape[0],3,x_down_10.shape[2],x_down_10.shape[3])


        for name,module in self.base_net._modules.items():
            # print(name)
            # print(module)
            
            # print(x_40_view.size())
            x_10_view=module(x_10_view)
            x_down_10_view=module(x_down_10_view)


            if name == self.layer_name:
                return x_10_view,x_down_10_view







class keyBlock_multi_Magnify_new(nn.Module):
    def __init__(self):
        super(keyBlock_multi_Magnify_new, self).__init__()

        ##定义参数a、b、c、d作为模型的可训练参数，并使用随机初始化
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))
        self.c = nn.Parameter(torch.randn(1))
        self.d = nn.Parameter(torch.randn(1))

        
        self.FeaExtractor_Layer1 = FeatureExtractor(layer_name='layer1')
        self.FeaExtractor_Layer2 = FeatureExtractor(layer_name='layer2')
        self.FeaExtractor_Layer3 = FeatureExtractor(layer_name='layer3')
        self.FeaExtractor_Layer4 = FeatureExtractor(layer_name='layer4')

       

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

       

    def forward(self,x_10,x_down_10):


        # 使用softmax函数对a、b、c、d进行归一化，使它们之和为1
        parameters = torch.cat([self.a, self.b, self.c, self.d])
        normalized_parameters = torch.softmax(parameters, dim=0)
        
       
        ###fea_layer1_10:(batch,64,64,64)
        ####f_layer1_down10:(batch,64,32,32)
        fea_layer1_10 ,f_layer1_down10 =  self.FeaExtractor_Layer1(x_10,x_down_10)
        
        ##f_layer2_40:(batch,128,32,32)
        ###fea_layer2_10:(batch,128,32,32)
        ####f_layer2_down10:(batch,128,16,16)
        fea_layer2_10 ,f_layer2_down10 =  self.FeaExtractor_Layer2(x_10,x_down_10)
        ##f_layer3_40:(batch,256,16,16)
        ###fea_layer3_10:(batch,256,16,16)
        ####f_layer3_down10:(batch,256,8,8)
        fea_layer3_10 ,f_layer3_down10 =  self.FeaExtractor_Layer3(x_10,x_down_10)
        ##f_layer4_40:(batch,512,8,8)
        ###fea_layer4_10:(batch,512,8,8)
        ####f_layer4_down10:(batch,512,4,4)
        fea_layer4_10 ,f_layer4_down10 =  self.FeaExtractor_Layer4(x_10,x_down_10)

   
        ###10倍降采样4倍的128*128的图块进行转置卷积(上采样）变成与10同样的tensor大小
        
        f_layer1_d10_inter =  F.interpolate(f_layer1_down10, size=64,mode='nearest')  
        f_layer2_d10_inter = F.interpolate(f_layer2_down10, size=32,mode='nearest')  
        f_layer3_d10_inter = F.interpolate(f_layer3_down10, size=16,mode='nearest')  
        f_layer4_d10_inter = F.interpolate(f_layer4_down10, size=8,mode='nearest')  


        #######f_layer1_d10_inter:(batch,64,64,64)
        ##进行3*3的维的卷积
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

        
      



        ##batch*64*1*1
        fea_merge_layer1 = self.avgpool(f_Lay1_10_mul_add)
        fea_merge_layer2 = self.avgpool(f_Lay2_10_mul_add)
        # print(fea_merge_layer2.size())
        fea_merge_layer3 = self.avgpool(f_Lay3_10_mul_add)
        fea_merge_layer4 = self.avgpool(f_Lay4_10_mul_add)

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
    
