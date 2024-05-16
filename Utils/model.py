import numpy as np
import torch.nn as nn
from torchvision import datasets, models, transforms
import os, torchvision, torch
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AttentionSlide_Batch(nn.Module):
    def __init__(self):
        super(AttentionSlide_Batch, self).__init__()
        self.L = 1000
        self.D = 128
        self.K = 1
        self.resnet = models.resnet34(pretrained=True)

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Linear(self.L*self.K, 1)

        self.flatten = nn.Flatten()

    def forward(self, x):
        x1 = x.view(x.shape[0]*x.shape[1], 3, x.shape[3], x.shape[4]) #squeeze batchsize for conv parallel.
        H = self.resnet(x1)
        A = self.attention(H)  # NxK
        # print(A.shape)
        A = A.view(x.shape[0], x.shape[1], -1)# recovery batchsize
        A = torch.transpose(A, 2, 1)  # KxN
        # print(A.shape)
        A = F.softmax(A, dim=2)  # softmax over N
        H = H.view(x.shape[0], x.shape[1], -1)
        M = torch.matmul(A, H)  # KxL attention to channel
        M = M.squeeze(dim=1)
        M = self.flatten(M)
        # print(M.shape)
        Y_prob = self.classifier(M)
        Y_prob = Y_prob.squeeze()
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A.squeeze()

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
    





class my_resnet_10x(nn.Module):
    def __init__(self):
        super(my_resnet_10x, self).__init__()
       
        self.resnet_10x = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.mlp = nn.Sequential(
            nn.Tanh(),
            nn.Linear(1000,64),
            # nn.Dropout(),
            nn.Tanh(),
            nn.Linear(64,2),
            # nn.Softmax(dim=1),
            
        )
    
    def forward(self, x_10):
        x_10 = x_10.view(x_10.shape[0],3,256,256)
        out_10x = self.resnet_10x(x_10)
        out_10x = self.mlp(out_10x) 
        out_10x = out_10x.view(x_10.shape[0],-1)
        # Y_prob = out_10x.view(x_10.shape[0],-1)


        Y_prob=F.softmax(out_10x,dim=1)
        _, Y_hat = torch.max(Y_prob, 1)

        # Y_prob=F.softmax(out_10x,dim=1)
        # Y_prob_1 = Y_prob[:,1]
        return Y_prob,Y_hat
    









    

# # 定义替换BatchNorm的LayerNorm模型

            
# class FeatureExtractor(nn.Module):
#     def __init__(self,layer_name):
#         super(FeatureExtractor, self).__init__()

#         # 加载预训练的ResNet-34模型
#         self.resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

#         modules_to_replace = [] 
#         for name, module in self.resnet.named_modules():
#             if isinstance(module, nn.BatchNorm2d):
#                 modules_to_replace.append((name, module)) 
#                 # 替换BatchNorm2d为LayerNorm，并在通道维度上进行归一化
#         for name, module in modules_to_replace: 
#             num_features = module.num_features 
#             layer_norm = nn.LayerNorm([num_features,]) 
#             setattr(self.resnet, name, layer_norm)

        
#         self.layer_name = layer_name
       
#     def forward(self,x_40, x_down_40,x_10):
#         ### x_40:[batch,bag(16),3,512,512]
#         ### x_down_40:[batch,bag(16),3,128,128]
#         ### x_10:[batch,1,3,512,512]
#         # print(x_40.size())
#         x_40_view = x_40.view(x_40.shape[0]*x_40.shape[1],3,x_40.shape[3],x_40.shape[4])

#         x_down_40_view = x_down_40.view(x_down_40.shape[0]*x_down_40.shape[1],3,x_down_40.shape[3],x_down_40.shape[4])

#         x_10 = x_10.squeeze(dim = 1)
# #         # print(x_10.size())
#         x_10_view = x_10.view(x_10.shape[0],3,x_10.shape[2],x_10.shape[3])
#         # print(x_40_view.size())
#         for name, module in self.resnet._modules.items():
            
#             x_40_view=module(x_40_view)
#             x_down_40_view = module(x_down_40_view)
#             x_10_view=module(x_10_view)

#             if name == self.layer_name:
#                 return x_40_view,x_down_40_view,x_10_view
            

# class FeatureExtractor(nn.Module):
#     def __init__(self,layer_name):
#         super(FeatureExtractor, self).__init__()

#         # 加载预训练的ResNet-34模型
#         self.resnet_40 = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
#         self.resnet_down_40 = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
#         self.resnet_10 = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

#         self.layer_name = layer_name
       
#     def forward(self, x_40,x_down_40,x_10):
#         ### x_40:[batch,bag(16),3,512,512]
#         ### x_down_40:[batch,bag(16),3,128,128]
#         ### x_10:[batch,1,3,512,512]
#         # print(x_40.size())
#         x_40_view = x_40.view(x_40.shape[0]*x_40.shape[1],3,x_40.shape[3],x_40.shape[4])

#         x_down_40_view = x_down_40.view(x_down_40.shape[0]*x_down_40.shape[1],3,x_down_40.shape[3],x_down_40.shape[4])

#         x_10 = x_10.squeeze(dim = 1)
# #         # print(x_10.size())
#         x_10_view = x_10.view(x_10.shape[0],3,x_10.shape[2],x_10.shape[3])

        

#         # print(x_40_view.size())
#         for name_40x, module_40x in self.resnet_40._modules.items():
            
#             if name_40x != 'fc' and name_40x != 'avgpool':
#                 if isinstance(module_40x, nn.BatchNorm2d) :
#                     # 替换BatchNorm层为LayerNorm层
#                     layer_norm = nn.LayerNorm([module_40x.num_features ,x_40_view.shape[2],x_40_view.shape[3]])
#                     layer_norm = layer_norm.to(device)
#                     x_40_view = layer_norm(x_40_view)
#                     # setattr(self.resnet_40, name_40x, layer_norm)
#                 else:
#                     x_40_view = module_40x(x_40_view)

#                 if name_40x == self.layer_name:
#                     x_40_output = x_40_view


#         for name_down_40x, module_down_40x in self.resnet_down_40._modules.items():
#             if name_down_40x != 'fc' and name_down_40x != 'avgpool':
#                 if isinstance(module_down_40x, nn.BatchNorm2d) :

#                     layer_norm = nn.LayerNorm([module_down_40x.num_features,x_down_40_view.shape[2],x_down_40_view.shape[3]])
#                     layer_norm = layer_norm.to(device)
#                     x_down_40_view = layer_norm(x_down_40_view)
                
#                 else:
#                     x_down_40_view=module_down_40x(x_down_40_view)

#                 if name_down_40x == self.layer_name:
#                     x_down_40_output = x_down_40_view


#         for name_10x, module_10x in self.resnet_10._modules.items():
#             if name_10x != 'fc' and name_10x != 'avgpool':

#                 if isinstance(module_10x, nn.BatchNorm2d) :
#                     layer_norm = nn.LayerNorm([module_10x.num_features,x_10_view.shape[2],x_10_view.shape[3]])
#                     layer_norm = layer_norm.to(device)
#                     x_10_view = layer_norm(x_10_view)
#                 else:
                
# #             # print(x_40_view.size())
#                     x_10_view=module_10x(x_10_view)

#                 if name_10x == self.layer_name:
#                     x_10_output = x_10_view

#         return x_40_output,x_down_40_output,x_10_output
            
        



class FeatureExtractor(nn.Module):
    def __init__(self,layer_name):
        super(FeatureExtractor, self).__init__()

        self.base_net = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        # self.base_net = ResNetLayerNorm()

        self.layer_name = layer_name
    def forward(self, x_40,x_down_40,x_10):
        ### x_40:[batch,bag(16),3,512,512]
        ### x_down_40:[batch,bag(16),3,128,128]
        ### x_10:[batch,1,3,512,512]
        # print(x_40.size())
        x_40_view = x_40.view(x_40.shape[0]*x_40.shape[1],3,x_40.shape[3],x_40.shape[4])
        # print(x_40_view.size())
        x_down_40_view = x_down_40.view(x_down_40.shape[0]*x_down_40.shape[1],3,x_down_40.shape[3],x_down_40.shape[4])
        x_10 = x_10.squeeze(dim = 1)
        # print(x_10.size())
        x_10_view = x_10.view(x_10.shape[0],3,x_10.shape[2],x_10.shape[3])

        for name,module in self.base_net._modules.items():
            # print(name)
            # print(module)
            x_40_view=module(x_40_view)
            # print(x_40_view.size())
            x_down_40_view=module(x_down_40_view)
            x_10_view=module(x_10_view)

            if name == self.layer_name:
                return x_40_view,x_down_40_view,x_10_view




class Fea_Layer1(nn.Module):
    def __init__(self):
        super(Fea_Layer1, self).__init__()
        
        self.FeaExtractor = FeatureExtractor(layer_name='layer1')
        
        # self.FeaExtractor_40x = FeatureExtractor_40x(layer_name='layer1')
        # self.FeaExtractor_down_40x = FeatureExtractor_40x_down(layer_name='layer1')
        # self.FeaExtractor_10x = FeatureExtractor_10x(layer_name='layer1')


        # self.pool_40 = nn.AdaptiveAvgPool3d((1,128,128))
        # self.pool_down_40 = nn.AdaptiveAvgPool3d((1,32,32))
        self.conv = nn.Conv3d(16, 1, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv_attention = nn.Conv3d(in_channels=64, out_channels=1, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))

    def forward(self, x_40,x_down_40,x_10):
        ####(batch*bag,64,128,128)
        ###(batch*bag,64,32,32)
        ###(batch,64,128,128)
        fea_layer1_40,fea_layer1_down40,fea_layer1_10 = self.FeaExtractor(x_40,x_down_40,x_10)
        # fea_layer1_40 = self.FeaExtractor_40x(x_40)
        # fea_layer1_down40 = self.FeaExtractor_down_40x(x_down_40)
        # fea_layer1_10 = self.FeaExtractor_10x(x_10)
        


        ##(batch,bag,64,128,128)
        fea_layer1_40_view = fea_layer1_40.view(x_40.shape[0],x_40.shape[1],fea_layer1_40.shape[1],fea_layer1_40.shape[2],fea_layer1_40.shape[3])
        ##(batch,bag,64,32,32)
        fea_layer1_down40_view = fea_layer1_down40.view(x_down_40.shape[0],x_down_40.shape[1],fea_layer1_down40.shape[1],fea_layer1_down40.shape[2],fea_layer1_down40.shape[3])
        


        # #######################attention
        # ##(batch,64,16,128,128)
        # fea_layer1_40_att = fea_layer1_40.view(x_40.shape[0],fea_layer1_40.shape[1],x_40.shape[1],fea_layer1_40.shape[2],fea_layer1_40.shape[3])
        # ##(batch,64,16,32,32)
        # fea_layer1_down40_att = fea_layer1_down40.view(x_down_40.shape[0],fea_layer1_down40.shape[1],x_down_40.shape[1],fea_layer1_down40.shape[2],fea_layer1_down40.shape[3])
        
        # ##(batch,1,16,128,128)
        # fea_layer1_40_att = self.conv_attention(fea_layer1_40_att)
        # ##batch,1,16,32,32
        # fea_layer1_down40_att = self.conv_attention(fea_layer1_down40_att)



        # ##pernute
        # ###(batch,128,128,16,64)
        # fea_layer1_40_view_Trans = torch.permute(fea_layer1_40_view, [0,3,4,1,2]) 
        # ###(batch,32,32,16,64)
        # fea_layer1_down40_view_Trans = torch.permute(fea_layer1_down40_view, [0,3,4,1,2]) 

        # ##(batch,128,128,1,16)
        # fea_layer1_40_att_Trans = torch.permute(fea_layer1_40_att, [0,3,4,1,2]) 
        # ##(batch,32,32,1,16)
        # fea_layer1_down40_att_Trans = torch.permute(fea_layer1_down40_att, [0,3,4,1,2]) 


        # ##softmax
        # ###(batch,128,128,1,16)
        # fea_layer1_40_att_Trans_softmax = F.softmax(fea_layer1_40_att_Trans, dim=4)
        # #(batch,32,32,1,16)  
        # fea_layer1_down40_att_Trans_softmax = F.softmax(fea_layer1_down40_att_Trans, dim=4)  

        # ### mul
        # ##(batch,128,128,1,64)
        # fea_layer1_40_att_mul = torch.matmul(fea_layer1_40_att_Trans_softmax,fea_layer1_40_view_Trans)
        # ##(batch,32,32,1,64)
        # fea_layer1_down40_att_mul = torch.matmul(fea_layer1_down40_att_Trans_softmax,fea_layer1_down40_view_Trans)


        
        # ###(batch,1,64,128,128)
        # f_layer1_40 = torch.permute(fea_layer1_40_att_mul,[0,3,4,1,2])
        # f_layer1_down40 = torch.permute(fea_layer1_down40_att_mul,[0,3,4,1,2])


        ##(batch,1,64,128,128)
        f_layer1_40 = self.conv(fea_layer1_40_view)
        ##batch,1,64,32,32
        f_layer1_down40 = self.conv(fea_layer1_down40_view)


        ##(batch,64,128,128)
        f_layer1_40 = f_layer1_40.squeeze(dim = 1)
        ##(batch,64,32,32)
        f_layer1_down40 = f_layer1_down40.squeeze(dim = 1)

        return f_layer1_40,f_layer1_down40,fea_layer1_10


class Fea_Layer2(nn.Module):
    def __init__(self):
        super(Fea_Layer2, self).__init__()
        
        self.FeaExtractor = FeatureExtractor(layer_name='layer2')
        # self.FeaExtractor_40x = FeatureExtractor_40x(layer_name='layer2')
        # self.FeaExtractor_down_40x = FeatureExtractor_40x_down(layer_name='layer2')
        # self.FeaExtractor_10x = FeatureExtractor_10x(layer_name='layer2')

        # self.pool_40 = nn.AdaptiveAvgPool3d((1,64,64))
        # self.pool_down_40 = nn.AdaptiveAvgPool3d((1,16,16))

        self.conv = nn.Conv3d(16, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_attention = nn.Conv3d(in_channels=128, out_channels=1, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))

    def forward(self, x_40,x_down_40,x_10):
        ####(batch*bag,128,64,64)
        ###(batch*bag,128,16,16)
        ###(batch,128,64,64)
        fea_layer2_40,fea_layer2_down40,fea_layer2_10 = self.FeaExtractor(x_40,x_down_40,x_10)
        # fea_layer2_40 = self.FeaExtractor_40x(x_40)
        # fea_layer2_down40 = self.FeaExtractor_down_40x(x_down_40)
        # fea_layer2_10 = self.FeaExtractor_10x(x_10)
        

        ##(batch,bag,128,64,64)
        fea_layer2_40_view = fea_layer2_40.view(x_40.shape[0],x_40.shape[1],fea_layer2_40.shape[1],fea_layer2_40.shape[2],fea_layer2_40.shape[3])
        ##(batch,bag,128,16,16)
        fea_layer2_down40_view = fea_layer2_down40.view(x_down_40.shape[0],x_down_40.shape[1],fea_layer2_down40.shape[1],fea_layer2_down40.shape[2],fea_layer2_down40.shape[3])
        
        #  #######################attention################
        # ##(batch,64,16,128,128)
        # fea_layer2_40_att = fea_layer2_40.view(x_40.shape[0],fea_layer2_40.shape[1],x_40.shape[1],fea_layer2_40.shape[2],fea_layer2_40.shape[3])
        # ##(batch,64,16,32,32)
        # fea_layer2_down40_att = fea_layer2_down40.view(x_down_40.shape[0],fea_layer2_down40.shape[1],x_down_40.shape[1],fea_layer2_down40.shape[2],fea_layer2_down40.shape[3])
        
        # ##(batch,1,16,128,128)
        # fea_layer2_40_att = self.conv_attention(fea_layer2_40_att)
        # ##batch,1,16,32,32
        # fea_layer2_down40_att = self.conv_attention(fea_layer2_down40_att)



        # ##pernute
        # ###(batch,128,128,16,64)
        # fea_layer2_40_view_Trans = torch.permute(fea_layer2_40_view, [0,3,4,1,2]) 
        # ###(batch,32,32,16,64)
        # fea_layer2_down40_view_Trans = torch.permute(fea_layer2_down40_view, [0,3,4,1,2]) 

        # ##(batch,128,128,1,16)
        # fea_layer2_40_att_Trans = torch.permute(fea_layer2_40_att, [0,3,4,1,2]) 
        # ##(batch,32,32,1,16)
        # fea_layer2_down40_att_Trans = torch.permute(fea_layer2_down40_att, [0,3,4,1,2]) 


        # ##softmax
        # ###(batch,128,128,1,16)
        # fea_layer2_40_att_Trans_softmax = F.softmax(fea_layer2_40_att_Trans, dim=4)
        # #(batch,32,32,1,16)  
        # fea_layer2_down40_att_Trans_softmax = F.softmax(fea_layer2_down40_att_Trans, dim=4)  

        # ### mul
        # ##(batch,128,128,1,64)
        # fea_layer2_40_att_mul = torch.matmul(fea_layer2_40_att_Trans_softmax,fea_layer2_40_view_Trans)
        # ##(batch,32,32,1,64)
        # fea_layer2_down40_att_mul = torch.matmul(fea_layer2_down40_att_Trans_softmax,fea_layer2_down40_view_Trans)


        
        # ###(batch,1,64,128,128)
        # f_layer2_40 = torch.permute(fea_layer2_40_att_mul,[0,3,4,1,2])
        # f_layer2_down40 = torch.permute(fea_layer2_down40_att_mul,[0,3,4,1,2])


      

        # ##(batch,64,128,128)
        # f_layer2_40 = f_layer2_40.squeeze(dim = 1)
        # ##(batch,64,32,32)
        # f_layer2_down40 = f_layer2_down40.squeeze(dim = 1)

        ##(batch,1,128,64,64)
        f_layer2_40 = self.conv(fea_layer2_40_view)
        ##batch,1,128,16,16
        f_layer2_down40 = self.conv(fea_layer2_down40_view)


        ##(batch,128,64,64)
        f_layer2_40 = f_layer2_40.squeeze(dim = 1)
        ##(batch,128,16,16)
        f_layer2_down40 = f_layer2_down40.squeeze(dim = 1)

        return f_layer2_40,f_layer2_down40,fea_layer2_10




class Fea_Layer3(nn.Module):
    def __init__(self):
        super(Fea_Layer3, self).__init__()
        
        self.FeaExtractor = FeatureExtractor(layer_name='layer3')
        # self.FeaExtractor_40x = FeatureExtractor_40x(layer_name='layer3')
        # self.FeaExtractor_down_40x = FeatureExtractor_40x_down(layer_name='layer3')
        # self.FeaExtractor_10x = FeatureExtractor_10x(layer_name='layer3')
        # self.pool_40 = nn.AdaptiveAvgPool3d((1,32,32))
        # self.pool_down_40 = nn.AdaptiveAvgPool3d((1,8,8))

        self.conv = nn.Conv3d(16, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_attention = nn.Conv3d(in_channels=256, out_channels=1, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))

    def forward(self, x_40,x_down_40,x_10):
        ####(batch*bag,256,32,32)
        ###(batch*bag,256,8,8)
        ###(batch,256,32,32)
        fea_layer3_40,fea_layer3_down40,fea_layer3_10 = self.FeaExtractor(x_40,x_down_40,x_10)
        # fea_layer3_40 = self.FeaExtractor_40x(x_40)
        # fea_layer3_down40 = self.FeaExtractor_down_40x(x_down_40)
        # fea_layer3_10 = self.FeaExtractor_10x(x_10)

        ##(batch,bag,256,32,32)
        fea_layer3_40_view = fea_layer3_40.view(x_40.shape[0],x_40.shape[1],fea_layer3_40.shape[1],fea_layer3_40.shape[2],fea_layer3_40.shape[3])
        ##(batch,bag,256,8,8)
        fea_layer3_down40_view = fea_layer3_down40.view(x_down_40.shape[0],x_down_40.shape[1],fea_layer3_down40.shape[1],fea_layer3_down40.shape[2],fea_layer3_down40.shape[3])
        
        
        
        #  #######################attention
        # ##(batch,64,16,128,128)
        # fea_layer3_40_att = fea_layer3_40.view(x_40.shape[0],fea_layer3_40.shape[1],x_40.shape[1],fea_layer3_40.shape[2],fea_layer3_40.shape[3])
        # ##(batch,64,16,32,32)
        # fea_layer3_down40_att = fea_layer3_down40.view(x_down_40.shape[0],fea_layer3_down40.shape[1],x_down_40.shape[1],fea_layer3_down40.shape[2],fea_layer3_down40.shape[3])
        
        # ##(batch,1,16,128,128)
        # fea_layer3_40_att = self.conv_attention(fea_layer3_40_att)
        # ##batch,1,16,32,32
        # fea_layer3_down40_att = self.conv_attention(fea_layer3_down40_att)



        # ##pernute
        # ###(batch,128,128,16,64)
        # fea_layer3_40_view_Trans = torch.permute(fea_layer3_40_view, [0,3,4,1,2]) 
        # ###(batch,32,32,16,64)
        # fea_layer3_down40_view_Trans = torch.permute(fea_layer3_down40_view, [0,3,4,1,2]) 

        # ##(batch,128,128,1,16)
        # fea_layer3_40_att_Trans = torch.permute(fea_layer3_40_att, [0,3,4,1,2]) 
        # ##(batch,32,32,1,16)
        # fea_layer3_down40_att_Trans = torch.permute(fea_layer3_down40_att, [0,3,4,1,2]) 


        # ##softmax
        # ###(batch,128,128,1,16)
        # fea_layer3_40_att_Trans_softmax = F.softmax(fea_layer3_40_att_Trans, dim=4)
        # #(batch,32,32,1,16)  
        # fea_layer3_down40_att_Trans_softmax = F.softmax(fea_layer3_down40_att_Trans, dim=4)  

        # ### mul
        # ##(batch,128,128,1,64)
        # fea_layer3_40_att_mul = torch.matmul(fea_layer3_40_att_Trans_softmax,fea_layer3_40_view_Trans)
        # ##(batch,32,32,1,64)
        # fea_layer3_down40_att_mul = torch.matmul(fea_layer3_down40_att_Trans_softmax,fea_layer3_down40_view_Trans)


        
        # ###(batch,1,64,128,128)
        # f_layer3_40 = torch.permute(fea_layer3_40_att_mul,[0,3,4,1,2])
        # f_layer3_down40 = torch.permute(fea_layer3_down40_att_mul,[0,3,4,1,2])



        ##(batch,1,256,32,32)
        f_layer3_40 = self.conv(fea_layer3_40_view)
        ##batch,1,256,8,8
        f_layer3_down40 = self.conv(fea_layer3_down40_view)

        ##(batch,256,32,32)
        f_layer3_40 = f_layer3_40.squeeze(dim = 1)
        ##(batch,256,8,8)
        f_layer3_down40 = f_layer3_down40.squeeze(dim = 1)

        return f_layer3_40,f_layer3_down40,fea_layer3_10




class Fea_Layer4(nn.Module):
    def __init__(self):
        super(Fea_Layer4, self).__init__()
        
        self.FeaExtractor = FeatureExtractor(layer_name='layer4')
        # self.FeaExtractor_40x = FeatureExtractor_40x(layer_name='layer4')
        # self.FeaExtractor_down_40x = FeatureExtractor_40x_down(layer_name='layer4')
        # self.FeaExtractor_10x = FeatureExtractor_10x(layer_name='layer4')
        # self.pool_40 = nn.AdaptiveAvgPool3d((1,16,16))
        # self.pool_down_40 = nn.AdaptiveAvgPool3d((1,4,4))

        self.conv = nn.Conv3d(16, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_attention = nn.Conv3d(in_channels=512, out_channels=1, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))

    def forward(self, x_40,x_down_40,x_10):
        ####(batch*bag,512,16,16)
        ###(batch*bag,512,4,4)
        ###(batch,512,16,16)
        fea_layer4_40,fea_layer4_down40,fea_layer4_10 = self.FeaExtractor(x_40,x_down_40,x_10)
        # fea_layer4_40 = self.FeaExtractor_40x(x_40)
        # fea_layer4_down40 = self.FeaExtractor_down_40x(x_down_40)
        # fea_layer4_10 = self.FeaExtractor_10x(x_10)

        ##(batch,bag,512,16,16)
        fea_layer4_40_view = fea_layer4_40.view(x_40.shape[0],x_40.shape[1],fea_layer4_40.shape[1],fea_layer4_40.shape[2],fea_layer4_40.shape[3])
        ##(batch,bag,512,4,4)
        fea_layer4_down40_view = fea_layer4_down40.view(x_down_40.shape[0],x_down_40.shape[1],fea_layer4_down40.shape[1],fea_layer4_down40.shape[2],fea_layer4_down40.shape[3])
        

        #  #######################attention
        # ##(batch,64,16,128,128)
        # fea_layer4_40_att = fea_layer4_40.view(x_40.shape[0],fea_layer4_40.shape[1],x_40.shape[1],fea_layer4_40.shape[2],fea_layer4_40.shape[3])
        # ##(batch,64,16,32,32)
        # fea_layer4_down40_att = fea_layer4_down40.view(x_down_40.shape[0],fea_layer4_down40.shape[1],x_down_40.shape[1],fea_layer4_down40.shape[2],fea_layer4_down40.shape[3])
        
        # ##(batch,1,16,128,128)
        # fea_layer4_40_att = self.conv_attention(fea_layer4_40_att)
        # ##batch,1,16,32,32
        # fea_layer4_down40_att = self.conv_attention(fea_layer4_down40_att)



        # ##pernute
        # ###(batch,128,128,16,64)
        # fea_layer4_40_view_Trans = torch.permute(fea_layer4_40_view, [0,3,4,1,2]) 
        # ###(batch,32,32,16,64)
        # fea_layer4_down40_view_Trans = torch.permute(fea_layer4_down40_view, [0,3,4,1,2]) 

        # ##(batch,128,128,1,16)
        # fea_layer4_40_att_Trans = torch.permute(fea_layer4_40_att, [0,3,4,1,2]) 
        # ##(batch,32,32,1,16)
        # fea_layer4_down40_att_Trans = torch.permute(fea_layer4_down40_att, [0,3,4,1,2]) 


        # ##softmax
        # ###(batch,128,128,1,16)
        # fea_layer4_40_att_Trans_softmax = F.softmax(fea_layer4_40_att_Trans, dim=4)
        # #(batch,32,32,1,16)  
        # fea_layer4_down40_att_Trans_softmax = F.softmax(fea_layer4_down40_att_Trans, dim=4)  

        # ### mul
        # ##(batch,128,128,1,64)
        # fea_layer4_40_att_mul = torch.matmul(fea_layer4_40_att_Trans_softmax,fea_layer4_40_view_Trans)
        # ##(batch,32,32,1,64)
        # fea_layer4_down40_att_mul = torch.matmul(fea_layer4_down40_att_Trans_softmax,fea_layer4_down40_view_Trans)


        
        # ###(batch,1,64,128,128)
        # f_layer4_40 = torch.permute(fea_layer4_40_att_mul,[0,3,4,1,2])
        # f_layer4_down40 = torch.permute(fea_layer4_down40_att_mul,[0,3,4,1,2])

        ##(batch,1,512,16,16)
        f_layer4_40 = self.conv(fea_layer4_40_view)
        ##batch,1,512,4,4
        f_layer4_down40 = self.conv(fea_layer4_down40_view)


        ##(batch,512,16,16)
        f_layer4_40 = f_layer4_40.squeeze(dim = 1)
        ##(batch,512,4,4)
        f_layer4_down40 = f_layer4_down40.squeeze(dim = 1)

        return f_layer4_40,f_layer4_down40,fea_layer4_10



# class keyBlock(nn.Module):
#     def __init__(self):
#         super(keyBlock, self).__init__()
        
#         self.fea_layer1 = Fea_Layer1()
#         self.fea_layer2 = Fea_Layer2()
#         self.fea_layer3 = Fea_Layer3()
#         self.fea_layer4 = Fea_Layer4()

#         self.inplanes_layer1 = 64  ##C通道数
#         self.inplanes_layer2 = 128  ##C通道数
#         self.inplanes_layer3 = 256  ##C通道数
#         self.inplanes_layer4 = 512  ##C通道数

        
#         self.conv1 = nn.Conv2d( in_channels = self.inplanes_layer1,out_channels = self.inplanes_layer1,kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv2 = nn.Conv2d( in_channels = self.inplanes_layer2,out_channels = self.inplanes_layer2,kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv3 = nn.Conv2d( in_channels = self.inplanes_layer3,out_channels = self.inplanes_layer3,kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv4 = nn.Conv2d( in_channels = self.inplanes_layer4,out_channels = self.inplanes_layer4,kernel_size=3, stride=1, padding=1, bias=False)

#         self.bn1 = nn.BatchNorm2d(self.inplanes_layer1 )
#         self.bn2 = nn.BatchNorm2d(self.inplanes_layer2 )
#         self.bn3 = nn.BatchNorm2d(self.inplanes_layer3 )
#         self.bn4 = nn.BatchNorm2d(self.inplanes_layer4 )


#         self.sigmoid = nn.Sigmoid()
#         self.relu = nn.ReLU()
#         self.elu = nn.ELU()

#         self.dense_layer1 = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=self.inplanes_layer1, out_channels=self.inplanes_layer1, kernel_size=4, stride=2, padding=1),
            
#         )
    
#         self.dense_layer2 = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=self.inplanes_layer2, out_channels=self.inplanes_layer2, kernel_size=4, stride=2, padding=1),

#         )
#         self.dense_layer3 = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=self.inplanes_layer3, out_channels=self.inplanes_layer3, kernel_size=4, stride=2, padding=1),

#         )
#         self.dense_layer4 = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=self.inplanes_layer4, out_channels=self.inplanes_layer4, kernel_size=4, stride=2, padding=1),

#         )

#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
#         self.fc_layer1 = nn.Linear(self.inplanes_layer1, 2)
#         self.fc_layer2 = nn.Linear(self.inplanes_layer2, 2)
#         self.fc_layer3 = nn.Linear(self.inplanes_layer3, 2)
#         self.fc_layer4 = nn.Linear(self.inplanes_layer4, 2)

#         self.fc_layer = nn.Sequential(
#             nn.Linear(960,64),
#             nn.Tanh(),
#             nn.Linear(64,2)
#         )

#     def forward(self, x_40,x_down_40,x_10):
#         ##f_layer1_40:(batch,64,64,64)
#         ####f_layer1_down40:(batch,64,32,32)
#         ###fea_layer1_10:(batch,64,64,64)
#         f_layer1_40,f_layer1_down40,fea_layer1_10 = self.fea_layer1(x_40,x_down_40,x_10)
#         ##f_layer2_40:(batch,128,32,32)
#         ####f_layer2_down40:(batch,128,16,16)
#         ###fea_layer2_10:(batch,128,64,64)
#         f_layer2_40,f_layer2_down40,fea_layer2_10 = self.fea_layer2(x_40,x_down_40,x_10)
#         ##f_layer3_40:(batch,256,32,32)
#         ####f_layer3_down40:(batch,256,8,8)
#         ###fea_layer3_10:(batch,256,32,32)
#         f_layer3_40,f_layer3_down40,fea_layer3_10 = self.fea_layer3(x_40,x_down_40,x_10)

#         ##f_layer4_40:(batch,512,16,16)
#         ####f_layer4_down40:(batch,512,4,4)
#         ###fea_layer4_10:(batch,512,16,16)
#         f_layer4_40,f_layer4_down40,fea_layer4_10 = self.fea_layer4(x_40,x_down_40,x_10)

#         #######f_layer1_d40_inter:(batch,64,128,128)
#         # ###40倍降采样4倍的128*128的图块进行最近邻插值变成与40同样的tensor大小
#         # f_layer1_d40_inter = F.interpolate(f_layer1_down40,size=[128, 128], mode="nearest")
#         # f_layer2_d40_inter = F.interpolate(f_layer2_down40,size=[64, 64], mode="nearest")
#         # f_layer3_d40_inter = F.interpolate(f_layer3_down40,size=[32, 32], mode="nearest")
#         # f_layer4_d40_inter = F.interpolate(f_layer4_down40,size=[16, 16], mode="nearest")



#         ###40倍降采样4倍的128*128的图块进行转置卷积(上采样）变成与40同样的tensor大小
#         f_layer1_d40_inter = self.dense_layer1(f_layer1_down40)
#         f_layer2_d40_inter = self.dense_layer2(f_layer2_down40)
#         f_layer3_d40_inter = self.dense_layer3(f_layer3_down40)
#         f_layer4_d40_inter = self.dense_layer4(f_layer4_down40)


#         #######f_layer1_d40_inter:(batch,64,128,128)
#         ##反卷积后进行3*3的维的卷积
#         f_layer1_d40_inter = self.sigmoid(self.bn1(self.conv1(f_layer1_d40_inter)))##维度不变
#         f_layer2_d40_inter = self.sigmoid(self.bn2(self.conv2(f_layer2_d40_inter)))##维度不变
#         f_layer3_d40_inter = self.sigmoid(self.bn3(self.conv3(f_layer3_d40_inter)))##维度不变
#         f_layer4_d40_inter = self.sigmoid(self.bn4(self.conv4(f_layer4_d40_inter)))##维度不变

#         ##f_layer1_40:(batch,64,128,128)
#         ##40倍的也进行3的卷积
#         f_layer1_40_ = self.sigmoid(self.bn1(self.conv1(f_layer1_40)))##维度不变
#         f_layer2_40_ = self.sigmoid(self.bn2(self.conv2(f_layer2_40)))##维度不变
#         f_layer3_40_ = self.sigmoid(self.bn3(self.conv3(f_layer3_40)))##维度不变
#         f_layer4_40_ = self.sigmoid(self.bn4(self.conv4(f_layer4_40)))##维度不变



#         ##点乘##(batch,256,128,128)
#         ##40倍和40倍降采样的进行点乘
#         f_Lay1_40_mul = torch.mul(f_layer1_d40_inter,f_layer1_40_)
#         f_Lay2_40_mul = torch.mul(f_layer2_d40_inter,f_layer2_40_)
#         f_Lay3_40_mul = torch.mul(f_layer3_d40_inter,f_layer3_40_)
#         f_Lay4_40_mul = torch.mul(f_layer4_d40_inter,f_layer4_40_)


#         ##点乘后在进行3的卷积
#         f_Lay1_40_mul = self.relu(self.bn1(self.conv1(f_Lay1_40_mul)))##维度不变
#         f_Lay2_40_mul = self.relu(self.bn2(self.conv2(f_Lay2_40_mul)))##维度不变
#         f_Lay3_40_mul = self.relu(self.bn3(self.conv3(f_Lay3_40_mul)))##维度不变
#         f_Lay4_40_mul = self.relu(self.bn4(self.conv4(f_Lay4_40_mul)))##维度不变

#         ##将点乘后（融合40倍和40倍降采样四倍的）在加上原来的40倍的特征
#         f_Lay1_40_mul_add = f_layer1_40 + f_Lay1_40_mul
#         f_Lay2_40_mul_add = f_layer2_40 + f_Lay2_40_mul
#         f_Lay3_40_mul_add = f_layer3_40 + f_Lay3_40_mul
#         f_Lay4_40_mul_add = f_layer4_40 + f_Lay4_40_mul


# ####### 融合后进行3*3的卷积
#         f_Lay1_40_mul_add = self.sigmoid(self.bn1(self.conv1(f_Lay1_40_mul_add)))##维度不变
#         f_Lay2_40_mul_add = self.sigmoid(self.bn2(self.conv2(f_Lay2_40_mul_add)))##维度不变
#         f_Lay3_40_mul_add = self.sigmoid(self.bn3(self.conv3(f_Lay3_40_mul_add)))##维度不变
#         f_Lay4_40_mul_add = self.sigmoid(self.bn4(self.conv4(f_Lay4_40_mul_add)))##维度不变

        
#         ##处理10倍的，只进行3的卷积
#         fea_layer1_10 = self.sigmoid(self.bn1(self.conv1(fea_layer1_10)))##维度不变
#         fea_layer2_10 = self.sigmoid(self.bn2(self.conv2(fea_layer2_10)))##维度不变
#         fea_layer3_10 = self.sigmoid(self.bn3(self.conv3(fea_layer3_10)))##维度不变
#         fea_layer4_10 = self.sigmoid(self.bn4(self.conv4(fea_layer4_10)))##维度不变

#         ###batch*64*128*128
#         fea_merge_layer1 = f_Lay1_40_mul_add + fea_layer1_10
#         fea_merge_layer2 = f_Lay2_40_mul_add + fea_layer2_10
#         fea_merge_layer3 = f_Lay3_40_mul_add + fea_layer3_10
#         fea_merge_layer4 = f_Lay4_40_mul_add + fea_layer4_10


#         ###batch*64*128*128
#         fea_merge_layer1 = self.relu(self.bn1(self.conv1(fea_merge_layer1)))##维度不变
#         fea_merge_layer2 = self.relu(self.bn2(self.conv2(fea_merge_layer2)))##维度不变
#         fea_merge_layer3 = self.relu(self.bn3(self.conv3(fea_merge_layer3)))##维度不变
#         fea_merge_layer4 = self.relu(self.bn4(self.conv4(fea_merge_layer4)))##维度不变

#         ##batch*64*1*1
#         fea_merge_layer1 = self.avgpool(fea_merge_layer1)
#         fea_merge_layer2 = self.avgpool(fea_merge_layer2)
#         # print(fea_merge_layer2.size())
#         fea_merge_layer3 = self.avgpool(fea_merge_layer3)
#         fea_merge_layer4 = self.avgpool(fea_merge_layer4)

#         #batch*64
#         fea_merge_layer1 = torch.flatten(fea_merge_layer1, 1)
#         fea_merge_layer2 = torch.flatten(fea_merge_layer2, 1)
#         # print(fea_merge_layer2.size())
#         fea_merge_layer3 = torch.flatten(fea_merge_layer3, 1)
#         fea_merge_layer4 = torch.flatten(fea_merge_layer4, 1)

#         fea_all = torch.cat((fea_merge_layer1, fea_merge_layer2,fea_merge_layer3,fea_merge_layer4), dim=1)

#         ##batch*2
#         # out_layer1 = self.fc_layer1(fea_merge_layer1)
#         # out_layer2 = self.fc_layer2(fea_merge_layer2)
#         # # print(out_layer2.size())
#         # out_layer3 = self.fc_layer3(fea_merge_layer3)
#         # out_layer4 = self.fc_layer4(fea_merge_layer4)
#         out_layer = self.fc_layer(fea_all)

#         # ##batch*2
#         # Y_layer1_prob=F.softmax(out_layer1,dim=1)
#         # Y_layer2_prob=F.softmax(out_layer2,dim=1)
#         # Y_layer3_prob=F.softmax(out_layer3,dim=1)
#         # Y_layer4_prob=F.softmax(out_layer4,dim=1)
#         Y_prob=F.softmax(out_layer,dim=1)




#         # Y_prob = Y_layer1_prob + Y_layer2_prob + Y_layer3_prob + Y_layer4_prob
#         # Y_prob = 0.1*out_layer1 + 0.2*out_layer2 + 0.3*out_layer3 + 0.4*out_layer4


#         # _40, Y_40_hat = torch.max(Y_40_prob, 1)
#         # _10, Y_10_hat = torch.max(Y_10_prob, 1)
#         _, Y_hat = torch.max(Y_prob, 1)

#         return Y_prob,Y_hat

   
# class MyModel(nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()

#         # 定义参数a、b、c、d作为模型的可训练参数，并使用随机初始化
#         self.a = nn.Parameter(torch.randn(1))
#         self.b = nn.Parameter(torch.randn(1))
#         self.c = nn.Parameter(torch.randn(1))
#         self.d = nn.Parameter(torch.randn(1))

#     def forward(self,loss1, loss2, loss3, loss4):
#         # 使用softmax函数对a、b、c、d进行归一化，使它们之和为1
#         parameters = torch.cat([self.a, self.b, self.c, self.d])
#         normalized_parameters = torch.softmax(parameters, dim=0)
#         # 计算a*loss1 + b*loss2 + c*loss3 + d*loss4
#         result = normalized_parameters[0]*loss1 + normalized_parameters[1]*loss2 + normalized_parameters[2]*loss3 + normalized_parameters[3]*loss4

#         return result, normalized_parameters

    
class keyBlock_new(nn.Module):
    def __init__(self):
        super(keyBlock_new, self).__init__()

        # 定义参数a、b、c、d作为模型的可训练参数，并使用随机初始化
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

        self.dense_layer1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.inplanes_layer1, out_channels=self.inplanes_layer1, kernel_size=4, stride=2, padding=1),
            
        )
    
        self.dense_layer2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.inplanes_layer2, out_channels=self.inplanes_layer2, kernel_size=4, stride=2, padding=1),

        )
        self.dense_layer3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.inplanes_layer3, out_channels=self.inplanes_layer3, kernel_size=4, stride=2, padding=1),

        )
        self.dense_layer4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.inplanes_layer4, out_channels=self.inplanes_layer4, kernel_size=4, stride=2, padding=1),

        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
        # self.fc_layer1 = nn.Linear(self.inplanes_layer1, 2)
        # self.fc_layer2 = nn.Linear(self.inplanes_layer2, 2)
        # self.fc_layer3 = nn.Linear(self.inplanes_layer3, 2)
        # self.fc_layer4 = nn.Linear(self.inplanes_layer4, 2)

        self.fc_layer1 = nn.Sequential(
            nn.Linear(self.inplanes_layer1,8),
            nn.Tanh(),
            nn.Linear(8,2)
        )
        self.fc_layer2 = nn.Sequential(
            nn.Linear(self.inplanes_layer2,16),
            nn.Tanh(),
            nn.Linear(16,2)
        )
        self.fc_layer3 = nn.Sequential(
            nn.Linear(self.inplanes_layer3,32),
            nn.Tanh(),
            nn.Linear(32,2)
        )
        self.fc_layer4 = nn.Sequential(
            nn.Linear(self.inplanes_layer4,64),
            nn.Tanh(),
            nn.Linear(64,2)
        )

        # self.param = MyModel()


    def forward(self, x_40,x_down_40,x_10):


        # 使用softmax函数对a、b、c、d进行归一化，使它们之和为1
        parameters = torch.cat([self.a, self.b, self.c, self.d])
        normalized_parameters = torch.softmax(parameters, dim=0)
        
        ##f_layer1_40:(batch,64,64,64)
        ####f_layer1_down40:(batch,64,32,32)
        ###fea_layer1_10:(batch,64,64,64)
        f_layer1_40,f_layer1_down40,fea_layer1_10 = self.fea_layer1(x_40,x_down_40,x_10)
        ##f_layer2_40:(batch,128,32,32)
        ####f_layer2_down40:(batch,128,16,16)
        ###fea_layer2_10:(batch,128,64,64)
        f_layer2_40,f_layer2_down40,fea_layer2_10 = self.fea_layer2(x_40,x_down_40,x_10)
        ##f_layer3_40:(batch,256,32,32)
        ####f_layer3_down40:(batch,256,8,8)
        ###fea_layer3_10:(batch,256,32,32)
        f_layer3_40,f_layer3_down40,fea_layer3_10 = self.fea_layer3(x_40,x_down_40,x_10)

        ##f_layer4_40:(batch,512,16,16)
        ####f_layer4_down40:(batch,512,4,4)
        ###fea_layer4_10:(batch,512,16,16)
        f_layer4_40,f_layer4_down40,fea_layer4_10 = self.fea_layer4(x_40,x_down_40,x_10)

        #######f_layer1_d40_inter:(batch,64,128,128)
        # ###40倍降采样4倍的128*128的图块进行最近邻插值变成与40同样的tensor大小
        # f_layer1_d40_inter = F.interpolate(f_layer1_down40,size=[128, 128], mode="nearest")
        # f_layer2_d40_inter = F.interpolate(f_layer2_down40,size=[64, 64], mode="nearest")
        # f_layer3_d40_inter = F.interpolate(f_layer3_down40,size=[32, 32], mode="nearest")
        # f_layer4_d40_inter = F.interpolate(f_layer4_down40,size=[16, 16], mode="nearest")



        ###40倍降采样4倍的128*128的图块进行转置卷积(上采样）变成与40同样的tensor大小
        f_layer1_d40_inter = self.dense_layer1(f_layer1_down40)
        f_layer2_d40_inter = self.dense_layer2(f_layer2_down40)
        f_layer3_d40_inter = self.dense_layer3(f_layer3_down40)
        f_layer4_d40_inter = self.dense_layer4(f_layer4_down40)


        #######f_layer1_d40_inter:(batch,64,128,128)
        ##反卷积后进行3*3的维的卷积
        f_layer1_d40_inter = self.sigmoid(self.bn1(self.conv1(f_layer1_d40_inter)))##维度不变
        f_layer2_d40_inter = self.sigmoid(self.bn2(self.conv2(f_layer2_d40_inter)))##维度不变
        f_layer3_d40_inter = self.sigmoid(self.bn3(self.conv3(f_layer3_d40_inter)))##维度不变
        f_layer4_d40_inter = self.sigmoid(self.bn4(self.conv4(f_layer4_d40_inter)))##维度不变

        ##f_layer1_40:(batch,64,128,128)
        ##40倍的也进行3的卷积
        f_layer1_40_ = self.sigmoid(self.bn1(self.conv1(f_layer1_40)))##维度不变
        f_layer2_40_ = self.sigmoid(self.bn2(self.conv2(f_layer2_40)))##维度不变
        f_layer3_40_ = self.sigmoid(self.bn3(self.conv3(f_layer3_40)))##维度不变
        f_layer4_40_ = self.sigmoid(self.bn4(self.conv4(f_layer4_40)))##维度不变



        ##点乘##(batch,256,128,128)
        ##40倍和40倍降采样的进行点乘
        f_Lay1_40_mul = torch.mul(f_layer1_d40_inter,f_layer1_40_)
        f_Lay2_40_mul = torch.mul(f_layer2_d40_inter,f_layer2_40_)
        f_Lay3_40_mul = torch.mul(f_layer3_d40_inter,f_layer3_40_)
        f_Lay4_40_mul = torch.mul(f_layer4_d40_inter,f_layer4_40_)


        ##点乘后在进行3的卷积
        f_Lay1_40_mul = self.relu(self.bn1(self.conv1(f_Lay1_40_mul)))##维度不变
        f_Lay2_40_mul = self.relu(self.bn2(self.conv2(f_Lay2_40_mul)))##维度不变
        f_Lay3_40_mul = self.relu(self.bn3(self.conv3(f_Lay3_40_mul)))##维度不变
        f_Lay4_40_mul = self.relu(self.bn4(self.conv4(f_Lay4_40_mul)))##维度不变

        ##将点乘后（融合40倍和40倍降采样四倍的）在加上原来的40倍的特征
        f_Lay1_40_mul_add = f_layer1_40 + f_Lay1_40_mul
        f_Lay2_40_mul_add = f_layer2_40 + f_Lay2_40_mul
        f_Lay3_40_mul_add = f_layer3_40 + f_Lay3_40_mul
        f_Lay4_40_mul_add = f_layer4_40 + f_Lay4_40_mul


####### 融合后进行3*3的卷积
        f_Lay1_40_mul_add = self.sigmoid(self.bn1(self.conv1(f_Lay1_40_mul_add)))##维度不变
        f_Lay2_40_mul_add = self.sigmoid(self.bn2(self.conv2(f_Lay2_40_mul_add)))##维度不变
        f_Lay3_40_mul_add = self.sigmoid(self.bn3(self.conv3(f_Lay3_40_mul_add)))##维度不变
        f_Lay4_40_mul_add = self.sigmoid(self.bn4(self.conv4(f_Lay4_40_mul_add)))##维度不变

        
        ##处理10倍的，只进行3的卷积
        fea_layer1_10 = self.sigmoid(self.bn1(self.conv1(fea_layer1_10)))##维度不变
        fea_layer2_10 = self.sigmoid(self.bn2(self.conv2(fea_layer2_10)))##维度不变
        fea_layer3_10 = self.sigmoid(self.bn3(self.conv3(fea_layer3_10)))##维度不变
        fea_layer4_10 = self.sigmoid(self.bn4(self.conv4(fea_layer4_10)))##维度不变

        ###batch*64*128*128
        fea_merge_layer1 = f_Lay1_40_mul_add + fea_layer1_10
        fea_merge_layer2 = f_Lay2_40_mul_add + fea_layer2_10
        fea_merge_layer3 = f_Lay3_40_mul_add + fea_layer3_10
        fea_merge_layer4 = f_Lay4_40_mul_add + fea_layer4_10


        ###batch*64*128*128
        fea_merge_layer1 = self.relu(self.bn1(self.conv1(fea_merge_layer1)))##维度不变
        fea_merge_layer2 = self.relu(self.bn2(self.conv2(fea_merge_layer2)))##维度不变
        fea_merge_layer3 = self.relu(self.bn3(self.conv3(fea_merge_layer3)))##维度不变
        fea_merge_layer4 = self.relu(self.bn4(self.conv4(fea_merge_layer4)))##维度不变

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

        # fea_all = torch.cat((fea_merge_layer1, fea_merge_layer2,fea_merge_layer3,fea_merge_layer4), dim=1)

        #batch*2
        out_layer1 = self.fc_layer1(fea_merge_layer1)
        out_layer2 = self.fc_layer2(fea_merge_layer2)
        # print(out_layer2.size())
        out_layer3 = self.fc_layer3(fea_merge_layer3)
        out_layer4 = self.fc_layer4(fea_merge_layer4)
        # out_layer = self.fc_layer(fea_all)

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