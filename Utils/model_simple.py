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
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.base_net = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        # self.base_net = ResNetLayerNorm()
        self.conv = nn.Conv3d(16, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.layer_name = 'layer4'

        
    def forward(self, x_40,x_10):
        ### x_40:[batch,bag(16),3,512,512]
        ### x_10:[batch,1,3,512,512]
        
        x_40_view = x_40.view(x_40.shape[0]*x_40.shape[1],3,x_40.shape[3],x_40.shape[4])

        x_10 = x_10.squeeze(dim = 1)
        x_10_view = x_10.view(x_10.shape[0],3,x_10.shape[2],x_10.shape[3])
        
        for name,module in self.base_net._modules.items():
            # print(name)
            # print(module)
            x_40_view=module(x_40_view)
            # print(x_40_view.size())
            x_10_view=module(x_10_view)
            


            if name == self.layer_name:
                ##batch*16,512,8,8
                out_40x = x_40_view
                ##batch,512,8,8
                out_10x = x_10_view
                break
            

        
        ##batch,16,512,8,8
        out_40x = out_40x.view(x_40.shape[0],x_40.shape[1],out_40x.shape[1],out_40x.shape[2],out_40x.shape[3])
        ##batch,1,512,8,8
        out_40x = self.conv(out_40x)
        ##batch,512,8,8
        out_40x = out_40x.squeeze(dim = 1)

        
        return out_40x,out_10x










    ###muti head self-attention
##一层一个融合后的特征
###batch*64*64*64
class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=12,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
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

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
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
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
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
    def __init__(self, num_classes=2, embed_dim =1024,depth=12, num_heads=8, mlp_ratio=4.0, qkv_bias=True,
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



        
        self.fea_extra =  FeatureExtractor()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        
        # self.a = nn.Parameter(torch.randn(1))
        # self.b = nn.Parameter(torch.randn(1))
        


        self.num_classes = num_classes
      
        self.embed_dim = embed_dim  ##C通道数
        ##特征数量即C通道上的值
        # self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.num_tokens = 1

        ##H*W
        self.num_patches = 64
        

        self.my_cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
      

        ##不用管，vit用不到，默认为none
        ##nn.Parameter 将一个固定不可训练的tensor转换成可以训练的类型parameter
        
        self.my_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + self.num_tokens, self.embed_dim))
        

        self.pos_drop = nn.Dropout(p=drop_ratio)
        ##构建等差序列0-depth，在l个block中drop_path_ratio是递增的
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        
        ##transformer的l个堆叠的blocks，是列表
        self.my_blocks = nn.Sequential(*[
            Block(dim=self.embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])

        
        self.my_norm = norm_layer(embed_dim)


        
        self.has_logits = False
        


        self.pre_logits = nn.Sequential(OrderedDict([
                ("fc1", nn.Linear(self.embed_dim , 64)),
                ("act", nn.Tanh()),
                ("fc2", nn.Linear(64, 2)),
            ]))
        
        self.dist_token =  None
        # Classifier head(s)
        # self.my_head = nn.Linear(64, num_classes) if num_classes > 0 else nn.Identity()

        # self.head_dist = None
      
        # Weight init
        nn.init.trunc_normal_(self.my_pos_embed, std=0.02)
   
        nn.init.trunc_normal_(self.my_cls_token, std=0.02)
       

        self.apply(_init_vit_weights)

    def forward_features(self, x_40,x_10):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        ##batch,512,8,8
        fea_40x,fea_10x = self.fea_extra(x_40,x_10)

        ##batch,1024,8,8
        fea_merge = torch.cat((fea_40x,fea_10x),dim = 1)
    
        # ##batch,512,1,1
        # fea_resnet_40x = self.avgpool(fea_40x)
        # fea_resnet_10x = self.avgpool(fea_10x)


        # ##batch,512
        # fea_resnet_40x = fea_resnet_40x.squeeze()
        # fea_resnet_10x = fea_resnet_10x.squeeze()




        ###batch,8*8,512
        # fea_40x_view = fea_40x.view(fea_40x.shape[0], fea_40x.shape[2]*fea_40x.shape[3], fea_40x.shape[1] )
        # fea_10x_view = fea_10x.view(fea_10x.shape[0], fea_10x.shape[2]*fea_10x.shape[3], fea_10x.shape[1] )
        fea_merge_view = fea_merge.view(fea_merge.shape[0], fea_merge.shape[2]*fea_merge.shape[3], fea_merge.shape[1] )

        




        # [1, 1, 768] -> [B, 1, 768]
        # class_token_40x = self.my_cls_token.expand(fea_40x_view.shape[0], -1, -1)
        # class_token_10x = self.my_cls_token.expand(fea_10x_view.shape[0], -1, -1)
        class_token = self.my_cls_token.expand(fea_merge_view.shape[0], -1, -1)


        



       
        # x_40x = torch.cat((class_token_40x, fea_40x_view), dim=1)  # [B, 197, 768]
        # x_10x = torch.cat((class_token_10x, fea_10x_view), dim=1)  # [B, 197, 768]
        x_merge = torch.cat((class_token, fea_merge_view), dim=1)  # [B, 197, 768]

       

       
        # x_40x = self.pos_drop(x_40x + self.my_pos_embed)
        # x_10x = self.pos_drop(x_10x + self.my_pos_embed)
        x_merge = self.pos_drop(x_merge + self.my_pos_embed)

      

        # x40_blocks = self.my_blocks(x_40x)
        # x10_blocks = self.my_blocks(x_10x)
        x_merge = self.my_blocks(x_merge)


       
        

        ##[batch,65,512]
        # x40_blocks = self.my_norm(x40_blocks) 
        # x10_blocks = self.my_norm(x10_blocks) 
        x_merge = self.my_norm(x_merge) 



        ##batch,512
        # x40_class = x40_blocks[:, 0]
        # x10_class = x10_blocks[:, 0]
        x_merge_class = x_merge[:, 0]


        

        # merge_40 = x40_class + fea_resnet_40x + x10_class
        # merge_10 = x10_class + fea_resnet_10x + x40_class


        # merge_40 = torch.cat((x40_class,fea_resnet_40x,x10_class),dim = 1)
        # merge_10 = torch.cat((x10_class,fea_resnet_10x,x40_class),dim=1)



      
        if self.dist_token is None:
            # return self.pre_logits(merge_40), self.pre_logits(merge_10)
            return self.pre_logits(x_merge_class)
        

        
        

    def forward(self, x_40,x_10):
        # parameters = torch.cat([self.a, self.b])
        # normalized_parameters = torch.softmax(parameters, dim=0)
        
        # out_40_class,out_10_class = self.forward_features(x_40,x_10)
        out_merge_class= self.forward_features(x_40,x_10)


        
        # out_40_class = self.my_head(out_40_class)
        # out_10_class = self.my_head(out_10_class)


        
         # ##batch*2
        # Y_40_prob=F.softmax(out_40_class,dim=1)
        # Y_10_prob=F.softmax(out_10_class,dim=1)
       
        Y_prob=F.softmax(out_merge_class,dim=1)


        # 计算a*loss1 + b*loss2 + c*loss3 + d*loss4
        # Y_prob = normalized_parameters[0]*Y_40_prob + normalized_parameters[1]*Y_10_prob 

        # Y_prob = Y_layer1_prob +  Y_layer2_prob + Y_layer3_prob + Y_layer4_prob
        # Y_prob = self.param.a.item() * Y_layer1_prob + self.param.b.item() * Y_layer2_prob + self.param.c.item() * Y_layer3_prob + self.param.d.item() * Y_layer4_prob

        # Y_prob = 0.1*out_layer1 + 0.2*out_layer2 + 0.3*out_layer3 + 0.4*out_layer4


        # _40, Y_40_hat = torch.max(Y_40_prob, 1)
        # _10, Y_10_hat = torch.max(Y_10_prob, 1)
        _, Y_hat = torch.max(Y_prob, 1)

        return Y_prob,Y_hat








def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)



