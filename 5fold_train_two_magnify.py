from torch.utils.tensorboard import SummaryWriter
import torch, os, random, glob, json, datetime, itertools
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import  transforms
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from Utils.dataloader_bag import TrainDataset_Fold_two_Magnify_sliderandom_innov,ValDataset_Fold_two_Magnify_innov
# from Utils.model import  my_resnet_10x,keyBlock
from Utils.model_multiLayer_Magnify import  keyBlock_multi_Magnify_new

from Utils.train_utils import train_model,  make_folder_two_scale_slide,make_two_scale_test_folder,train_two_scale_model,train_two_Magnify_model
from torch.backends import cudnn as cudnn
random.seed(0)
plt.ion()   # interactive mode
now = datetime.datetime.now()
current_timestamp = now.strftime('%m-%d-%H_%M')

cudnn.benchmark = True
log_dir = "Z:/admin/LXH/lxh_train_code/mutation/Logs_Tumor_targeted_therapy/"+current_timestamp
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#############修改这里的bag_size,文件地址###############421032


batchsize = 128
# bag_size = 16
seed = 0
epoch = 20
norm_mean = [0.5, 0.5, 0.5]
norm_std = [0.5, 0.5, 0.5]
if not os.path.exists('Z:/admin/LXH/lxh_train_code/mutation/Checkpoint_targeted_surgery_two_Magnify_norm_balance_copy'):
    os.makedirs('Z:/admin/LXH/lxh_train_code/mutation/Checkpoint_targeted_surgery_two_Magnify_norm_balance_copy')

transform_original = {
    'train': transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ]),
    'val': transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ]),
}


transform_down = {
    'train': transforms.Compose([
        transforms.Resize((128,128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ]),
    'val': transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ]),
}



label_path = r'Z:/admin/LXH/lxh_train_code/patient_label_drug_resistance_or_sensitivenes_surgery_Notest.csv'
label_test_path = r'Z:/admin/LXH/lxh_train_code/patient_label_drug_resistance_or_sensitivenes_surgery_test.csv'
# label_tcga_test_path = r'Z:/admin/LXH/lxh_train_code/patient_label_drug_resistance_or_sensitivenes_surgery_tcga_test.csv'



data_dir_40x = 'Z:/admin/Xiaomin0335/HER2_BC/Tile_Filtered/Tumor/Tumor_Tiles_40x_512_Original_position_recut_Normalized'
data_dir_10x = 'Z:/admin/Xiaomin0335/HER2_BC/Tile_Filtered/Tumor/Tumor_Tiles_10x_512_Normalized'

# data_dir_40x = 'Z:/admin/LXH/tcga_tiles/tcga_40x_512_norm'
# data_dir_10x = 'Z:/admin/LXH/tcga_tiles/tcga_10x_512_norm'

labels = pd.read_csv(label_path)
test_labels = pd.read_csv(label_test_path)



event = labels["label"]
p_name = labels["slide_name"]

test_event = test_labels["label"]
test_p_name = test_labels["slide_name"]


test_folder = make_two_scale_test_folder(data_dir_10x,data_dir_40x, test_event, test_p_name)


skf = StratifiedKFold(n_splits=3, shuffle=True,random_state = seed)
rdn_index = skf.split(np.zeros(len(event)),event)
# weights = "Z:/admin/LXH/lxh_train_code/mutation/Utils/vit_base_patch16_224.pth"

#按Slide分组
print('Training: Split by Slide\n')
if len(os.listdir('Z:/admin/LXH/lxh_train_code/mutation/Val_list_surgery_two_Magnify_norm_balance')) < 6:

    for k_fold,(train_index, val_index) in enumerate(rdn_index) :
 
        train_folder, val_folder = make_folder_two_scale_slide(data_dir_10x,data_dir_40x, train_index, val_index, event,p_name, k_fold)
     


else:
    for i_fold in range (2,3):

        train_file = 'Z:/admin/LXH/lxh_train_code/mutation/Val_list_surgery_two_Magnify_norm_balance/train_folder'+str(i_fold)+'.json'
        with open(train_file, 'r') as f:
            train_folder_ = json.load(f)

        val_file = 'Z:/admin/LXH/lxh_train_code/mutation/Val_list_surgery_two_Magnify_norm_balance/val_folder'+str(i_fold)+'.json'
        with open(val_file, 'r') as f:
            val_folder_ = json.load(f)
    
        print('k_fold:',i_fold)
        image_datasets = {'train': TrainDataset_Fold_two_Magnify_sliderandom_innov(train_folder_, transform_original['train'], transform_down['train'],ratio=1), 'val': ValDataset_Fold_two_Magnify_innov(val_folder_,transform_original['train'], transform_down['train'],ratio=1)}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchsize, shuffle=True, drop_last = True) for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

        model_ft = keyBlock_multi_Magnify_new()



        
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
        
        
        # weights_dict = torch.load(weights, map_location=device)
       ## 删除不需要的权重
        # if model_ft.has_logits == 'False':
        #     del_keys = ['head.weight', 'head.bias','cls_token','pos_embed','blocks','norm.weight','norm.bias','head.weight','head.bias'] 
           
        #     for k in del_keys:
        #         del weights_dict[k]

       
        # print(model_ft.load_state_dict(weights_dict, strict=False))




        
        
        # for name, para in model_ft.named_parameters():
        #     # 除head, pre_logits外，其他权重全部冻结
        #     if "head_layer1" not in name and "pre_logits" not in name and "head_layer2" not in name and "head_layer3" not in name and "head_layer4" not in name and "blocks_layer1" not in name and "blocks_layer2" not in name and "blocks_layer3" not in name and "blocks_layer4" not in name:
        #         para.requires_grad_(False)
        #     else:
        #         print("training {}".format(name))

       
        model_ft = model_ft.to(device)
        criterion = nn.CrossEntropyLoss()

        for param in model_ft.parameters():
            param.requires_grad = True

        params = [p for p in model_ft.parameters() if p.requires_grad]
        optimizer_ft = optim.Adam(params, lr=0.00005)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, 'min', factor = 0.5,patience = 1)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_ft, milestones=[7,9], gamma=0.1)

        writer = SummaryWriter(log_dir+'/'+str(i_fold))
        model_ft = train_two_Magnify_model(model_ft, criterion, optimizer_ft, dataloaders, dataset_sizes, epoch, writer,scheduler)
        torch.save(model_ft.state_dict(), os.path.join('Z:/admin/LXH/lxh_train_code/mutation/Checkpoint_targeted_surgery_two_Magnify_norm_balance_copy', 'resnet34_two_Magnify_10_10down_lr5_drop_fold_' + str(i_fold) + '.pt'))


   
    # print('Slide validation\n')
    # wild = 0
    # brca = 0
    # wild_wrong = 0
    # brca_wrong = 0
    # val_folder_ = list(itertools.chain.from_iterable(val_folder))
    # for wsi in val_folder_:
    #     pred_aggregate = []
    #     # print(wsi)
    #     bag_image_dataset = TestDataset(wsi, data_transforms['val'], bag_size=bag_size)
    #     bag_image_dataloader = torch.utils.data.DataLoader(bag_image_dataset, batch_size = val_batchsize, shuffle=False, drop_last = True) 
    #     # bag_image = bag_image.unsqueeze(dim=0)
    #     # bag_image = bag_image.to(device)
    #     # print(bag_image.shape)
    #     model_ft.eval()
    #     with torch.no_grad():
    #         for bag_image in bag_image_dataloader:
    #             bag_image = bag_image.to(device)
    #             outputs, preds, A= model_ft(bag_image)
    #             outputs = outputs.cpu().numpy()
    #             preds = preds.cpu().numpy()
    #             pred_aggregate.append(preds)
    #             # print(preds)
    #     if 'WILD' in wsi:
    #         wild += 1
    #         pred_aggregate_ = list(itertools.chain.from_iterable(pred_aggregate))
    #         slide_result = np.mean(pred_aggregate_)
    #         # print(slide_result)
    #         if slide_result > 0.5:
    #             wild_wrong += 1
    #     else:
    #         brca += 1
    #         pred_aggregate_ = list(itertools.chain.from_iterable(pred_aggregate))
    #         slide_result = np.mean(pred_aggregate_)
    #         # print(slide_result)
    #         if slide_result < 0.5:
    #             brca_wrong += 1

    # print('fold: ',k_fold, 'wild: ',wild, 'brca: ',brca, 'wild_wrong: ',wild_wrong, 'brca_wrong: ', brca_wrong)

