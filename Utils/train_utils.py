import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import os, copy, torch, itertools, json, pypinyin, glob
from tqdm import tqdm, trange
from PIL import Image
from torchvision.io import read_image
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from Utils.caculate import accuracy
import re
import random
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs, writer, scheduler,use_amp = True):
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    running_step = 0
    early_stopped = 0.00001

    for epoch in range(num_epochs):

        # Each epoch has a training and validation phase
        for phase in ['train','val']:
            early_times = 0
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            with tqdm(dataloaders[phase]) as t:
                t.set_description("Epoch = {:d}/{:d}".format(epoch, num_epochs))
                for inputs, labels in t:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        with torch.cuda.amp.autocast(enabled=use_amp):
                            outputs, preds= model(inputs)
                            loss = criterion(outputs, labels.long())
                            acc =  accuracy(preds,labels)

                            writer.add_scalar(phase+'loss', loss.item(), running_step)
                            writer.add_scalar(phase+'batch_acc', acc.item(), running_step)
                            
                            running_step += 1
                            t.set_postfix(batch_loss=loss.item())
                            # backward + optimize only if in training phase
                            if phase == 'train':
                                scaler.scale(loss).backward()
                                scaler.step(optimizer)
                                scaler.update()
                                if loss.item() < early_stopped:
                                    early_times += 1
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    if early_times >= 10:
                        break

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            if  phase == 'train':
                epoch_trian_loss = epoch_loss

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print('Optimizer learning rate : {:.13f}'.format(optimizer.param_groups[0]['lr']))

        scheduler.step()

            

    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


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

    

def train_two_scale_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs, writer, scheduler,use_amp = True):
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    running_step = 0
    early_stopped = 0.00001
    # param_model = MyModel()
    
    for epoch in range(num_epochs):

        # Each epoch has a training and validation phase
        for phase in ['train','val']:
            early_times = 0
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            with tqdm(dataloaders[phase]) as t:
                t.set_description("Epoch = {:d}/{:d}".format(epoch, num_epochs))
                for inputs_40x,inputs_10x, inputs_down_10x,labels in t:

                    inputs_40x = inputs_40x.to(device)
                    inputs_10x = inputs_10x.to(device)
                    inputs_down_10x = inputs_down_10x.to(device)
                

                    labels = labels.to(device)
               
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        with torch.cuda.amp.autocast(enabled=use_amp):
                            # outputs1,outputs2,outputs3,outputs4, preds= model(inputs_40x,inputs_down_40x,inputs_10x)

                           

                            outputs, preds,outputs1,outputs2,outputs3,outputs4= model(inputs_40x,inputs_10x,inputs_down_10x)

                            loss = criterion(outputs, labels.long())
                            loss1 = criterion(outputs1, labels.long())
                            loss2 = criterion(outputs2, labels.long())
                            loss3 = criterion(outputs3, labels.long())
                            loss4 = criterion(outputs4, labels.long())



                            acc =  accuracy(preds,labels)
                            # loss, normalized_params = param_model(loss1, loss2, loss3, loss4)

                            writer.add_scalar(phase+'loss', loss.item(), running_step)
                            writer.add_scalar(phase+'layer1_loss', loss1.item(), running_step)
                            writer.add_scalar(phase+'layer2_loss', loss2.item(), running_step)
                            writer.add_scalar(phase+'layer3_loss', loss3.item(), running_step)
                            writer.add_scalar(phase+'layer4_loss', loss4.item(), running_step)

                            writer.add_scalar(phase+'batch_acc', acc.item(), running_step)
                            
                            running_step += 1
                            t.set_postfix(batch_loss=loss.item())
                            # backward + optimize only if in training phase
                            if phase == 'train':
                                scaler.scale(loss).backward()
                                scaler.step(optimizer)
                                scaler.update()
                                if loss.item() < early_stopped:
                                    early_times += 1
                    # statistics
                    running_loss += loss.item() * inputs_10x.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    if early_times >= 10:
                        break

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            writer.add_scalar(phase+'epoch_loss', epoch_loss, epoch)
            writer.add_scalar(phase+'epoch_acc', epoch_acc, epoch)


            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            if  phase == 'val':
                epoch_trian_loss = epoch_loss

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print('Optimizer learning rate : {:.13f}'.format(optimizer.param_groups[0]['lr']))

        scheduler.step(epoch_trian_loss)


    # # 打印最终的归一化参数
    # print("\nNormalized parameters:")
    # print("a =", normalized_params[0].item())
    # print("b =", normalized_params[1].item())
    # print("c =", normalized_params[2].item())
    # print("d =", normalized_params[3].item())

            

    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def train_two_Magnify_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs, writer, scheduler,use_amp = True):
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    running_step = 0
    early_stopped = 0.00001
    # param_model = MyModel()
    
    for epoch in range(num_epochs):

        # Each epoch has a training and validation phase
        for phase in ['train','val']:
            early_times = 0
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            with tqdm(dataloaders[phase]) as t:
                t.set_description("Epoch = {:d}/{:d}".format(epoch, num_epochs))
                for inputs_10x, inputs_down_10x,labels in t:

                    
                    inputs_10x = inputs_10x.to(device)
                    inputs_down_10x = inputs_down_10x.to(device)
                

                    labels = labels.to(device)
               
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        with torch.cuda.amp.autocast(enabled=use_amp):
                            # outputs1,outputs2,outputs3,outputs4, preds= model(inputs_40x,inputs_down_40x,inputs_10x)

                           

                            outputs, preds,outputs1,outputs2,outputs3,outputs4= model(inputs_10x,inputs_down_10x)

                            loss = criterion(outputs, labels.long())
                            loss1 = criterion(outputs1, labels.long())
                            loss2 = criterion(outputs2, labels.long())
                            loss3 = criterion(outputs3, labels.long())
                            loss4 = criterion(outputs4, labels.long())



                            acc =  accuracy(preds,labels)
                            # loss, normalized_params = param_model(loss1, loss2, loss3, loss4)

                            writer.add_scalar(phase+'loss', loss.item(), running_step)
                            writer.add_scalar(phase+'layer1_loss', loss1.item(), running_step)
                            writer.add_scalar(phase+'layer2_loss', loss2.item(), running_step)
                            writer.add_scalar(phase+'layer3_loss', loss3.item(), running_step)
                            writer.add_scalar(phase+'layer4_loss', loss4.item(), running_step)

                            writer.add_scalar(phase+'batch_acc', acc.item(), running_step)
                            
                            running_step += 1
                            t.set_postfix(batch_loss=loss.item())
                            # backward + optimize only if in training phase
                            if phase == 'train':
                                scaler.scale(loss).backward()
                                scaler.step(optimizer)
                                scaler.update()
                                if loss.item() < early_stopped:
                                    early_times += 1
                    # statistics
                    running_loss += loss.item() * inputs_10x.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    if early_times >= 10:
                        break

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            writer.add_scalar(phase+'epoch_loss', epoch_loss, epoch)
            writer.add_scalar(phase+'epoch_acc', epoch_acc, epoch)


            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            if  phase == 'val':
                epoch_trian_loss = epoch_loss

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print('Optimizer learning rate : {:.13f}'.format(optimizer.param_groups[0]['lr']))

        scheduler.step(epoch_trian_loss)


    # # 打印最终的归一化参数
    # print("\nNormalized parameters:")
    # print("a =", normalized_params[0].item())
    # print("b =", normalized_params[1].item())
    # print("c =", normalized_params[2].item())
    # print("d =", normalized_params[3].item())

            

    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def validation(model, criterion, dataloaders, dataset_sizes, use_amp = True):

    running_step = 0
    # Each epoch has a training and validation phase
    phase = 'val'
    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    with tqdm(dataloaders) as t:
        for inputs, labels in t:
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.set_grad_enabled(False):
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs, preds, A = model(inputs)
                    loss = criterion(outputs, labels.long())
                    running_step += 1
                    t.set_postfix(batch_loss=loss.item())
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_sizes
    epoch_acc = running_corrects.double() / dataset_sizes

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        phase, epoch_loss, epoch_acc))

def make_folder_list_slide(data_path,train_index, val_index, event, p_name, k_fold):
        ##p_name
        train_p_name = p_name.values[train_index]
        val_p_name = p_name.values[val_index]
        ##p_event /status
        train_p_event = event.values[train_index]
        train_p_event = train_p_event.tolist()
        val_p_event = event.values[val_index]
        val_p_event = val_p_event.tolist()


       

        t_pfolders = []
        v_pfolders = []
        ##每一折的训练和验证的路径
        for t_pname in train_p_name:
            t_pfolder = os.path.join(data_path,t_pname)
            t_pfolders.append(t_pfolder)
        for v_pname in val_p_name:
            v_pfolder = os.path.join(data_path,v_pname)
            v_pfolders.append(v_pfolder)

        train_folder = [t_pfolders,train_p_event]
        val_folder = [v_pfolders,val_p_event]

        if not os.path.exists('D:/lxh/python_learn/lxh_surv_code/mutation/Val_list_therapy_40x_norm_balance'):
            os.makedirs('D:/lxh/python_learn/lxh_surv_code/mutation/Val_list_therapy_40x_norm_balance')
        with open('D:/lxh/python_learn/lxh_surv_code/mutation/Val_list_therapy_40x_norm_balance/val_folder'+ str(k_fold) +'.json', 'w') as f:
            json.dump(val_folder, f)
        with open('D:/lxh/python_learn/lxh_surv_code/mutation/Val_list_therapy_40x_norm_balance/train_folder'+ str(k_fold) +'.json', 'w') as ft:
            json.dump(train_folder, ft)

        return train_folder, val_folder






def make_folder_two_scale_slide(data_path_10x,data_path_40x,train_index, val_index, event, p_name, k_fold):
        ##p_name
        train_p_name = p_name.values[train_index]
        val_p_name = p_name.values[val_index]
        ##p_event /status
        train_p_event = event.values[train_index]
        train_p_event = train_p_event.tolist()
        val_p_event = event.values[val_index]
        val_p_event = val_p_event.tolist()


       

        t_pfolders_10x = []
        t_pfolders_40x = []

        v_pfolders_10x = []
        v_pfolders_40x = []

        ##每一折的训练和验证的路径
        for t_pname in train_p_name:
            t_pfolder_10x = os.path.join(data_path_10x,t_pname)
            t_pfolder_40x = os.path.join(data_path_40x,t_pname)

            t_pfolders_10x.append(t_pfolder_10x)
            t_pfolders_40x.append(t_pfolder_40x)


        for v_pname in val_p_name:
            v_pfolder_10x = os.path.join(data_path_10x,v_pname)
            v_pfolder_40x = os.path.join(data_path_40x,v_pname)

            v_pfolders_10x.append(v_pfolder_10x)
            v_pfolders_40x.append(v_pfolder_40x)


        train_folder = [t_pfolders_10x,t_pfolders_40x,train_p_event]
        val_folder = [v_pfolders_10x,v_pfolders_40x,val_p_event]

        if not os.path.exists('Z:/admin/LXH/lxh_train_code/mutation/Val_list_surgery_two_Magnify_norm_balance'):
            os.makedirs('Z:/admin/LXH/lxh_train_code/mutation/Val_list_surgery_two_Magnify_norm_balance')
        with open('Z:/admin/LXH/lxh_train_code/mutation/Val_list_surgery_two_Magnify_norm_balance/val_folder'+ str(k_fold) +'.json', 'w') as f:
            json.dump(val_folder, f)
        with open('Z:/admin/LXH/lxh_train_code/mutation/Val_list_surgery_two_Magnify_norm_balance/train_folder'+ str(k_fold) +'.json', 'w') as ft:
            json.dump(train_folder, ft)

        return train_folder, val_folder




def make_Tiles_folder_train_val(data_path_10x,data_path_40x, event, p_name, bag_size,ratio = 1):
        ##p_name
        train_val_p_name = p_name.values
        
        ##p_event /status
        train_val_p_event = event.values
        train_val_p_event = train_val_p_event.tolist()
       

       

        list_pfolders_10x = []
        list_pfolders_40x = []

        

        ##每一折的训练和验证的路径
        for t_pname in train_val_p_name:
            pfolders_10x = os.path.join(data_path_10x,t_pname)
            pfolders_40x = os.path.join(data_path_40x,t_pname)

            list_pfolders_10x.append(pfolders_10x)
            list_pfolders_40x.append(pfolders_40x)


        
        train_val_folder = [list_pfolders_10x,list_pfolders_40x,train_val_p_event]
        

        ###tile数据
        train_val_Tiles_folder_10x = []
        train_val_Tiles_folder_40x = []
        train_val_Tiles_event = []
        ##10x
        for p_idx,p_path in enumerate(train_val_folder[0]):
            
            p_event = train_val_folder[2][p_idx]
            total_list = glob.glob(p_path+"/*.jpg")
            total_list.sort(key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
           

            for tile in total_list:
                # tile_chunk = [tile,p_event]
                train_val_Tiles_folder_10x.append(tile)
                train_val_Tiles_event.append(p_event)

        ###40x
        for p_idx,p_path in enumerate(train_val_folder[1]):
            
            # p_event = file_dirs[2][p_idx]
            total_list = glob.glob(p_path+"/*.jpg")
            total_list.sort(key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
           

            for tile in total_list:
                # tile_chunk = [tile,p_event]
                train_val_Tiles_folder_40x.append(tile)
        # Tiles_folder = list(itertools.chain.from_iterable(Tiles_folder))
        
        train_list_chunks_40x = [train_val_Tiles_folder_40x[x:x+bag_size] for x in range(0, len(train_val_Tiles_folder_40x)-bag_size, bag_size)]
        train_list_chunks_10x = train_val_Tiles_folder_10x
        train_list_Tiles_event = train_val_Tiles_event

        
        

        train_list_chunks_10x = train_list_chunks_10x[:int(len(train_list_chunks_10x)*ratio)]
        train_list_chunks_40x = train_list_chunks_40x[:int(len(train_list_chunks_40x)*ratio)]

        # random.shuffle(self.list_chunks)


        train_all_list = [] 

        for idx_bag,bag_tile in  enumerate(train_list_chunks_40x) :
            ##[[16*tile],tile,]
            tile_10_40 = [bag_tile , train_list_chunks_10x[idx_bag], train_list_Tiles_event[idx_bag]]
            train_all_list.append(tile_10_40)

        train_all_list = train_all_list[:int(len(train_all_list)*ratio)]

        ##训练的平衡数据
        list_chunk_0 = []
        list_chunk_1 = []

        for idx_chunk,i_chunk  in enumerate(train_all_list):
            if i_chunk[2] == 0:
                list_chunk_0.append(i_chunk)

            else:
                list_chunk_1.append(i_chunk)
        
        list_chunk_0_select = random.sample(list_chunk_0,k = int(len(list_chunk_1)*1.5))

        
        train_all_list_select = [list_chunk_0_select,list_chunk_1]
        train_all_list_select = list(itertools.chain.from_iterable(train_all_list_select))

        train_all_list_select = train_all_list_select
        train_all_list_select = train_all_list_select[:int(len(train_all_list_select)*ratio)]

       
        ##label

        label_all_list = [] 

        for idx_bag,tile_chunk in  enumerate(train_all_list_select) :
            ##[[16*tile],tile,]
            label_10_40 = tile_chunk[2]
            label_all_list.append(label_10_40)

        label_all_list = label_all_list[:int(len(label_all_list)*ratio)]



        np_train_all_list_select = np.array(train_all_list_select)
        np_label_all_list = np.array(label_all_list)


        return np_train_all_list_select, np_label_all_list



def make_tiles_folder_two_scale_slide(train_index, val_index, folder_tiles, k_fold,ratio = 1):
        ##p_name
        tiles_train_list = folder_tiles[train_index]
        tiles_train_list = tiles_train_list.tolist()

        tiles_vaild_list = folder_tiles[val_index]
        tiles_vaild_list = tiles_vaild_list.tolist()

        #  ##训练的平衡数据
        # list_chunk_0 = []
        # list_chunk_1 = []

        # for idx_chunk,i_chunk  in enumerate(tiles_train_list):
        #     if i_chunk[2] == 0:
        #         list_chunk_0.append(i_chunk)

        #     else:
        #         list_chunk_1.append(i_chunk)
        
        # list_chunk_0_select = random.sample(list_chunk_0,k = int(len(list_chunk_1)*1.5))

        
        # train_all_list_select = [list_chunk_0_select,list_chunk_1]
        # train_all_list_select = list(itertools.chain.from_iterable(train_all_list_select))

        # train_all_list_select = train_all_list_select
        # train_all_list_select = train_all_list_select[:int(len(train_all_list_select)*ratio)]

        # random.shuffle(train_all_list_select)
        # random.shuffle(tiles_vaild_list)


        if not os.path.exists('Z:/admin/LXH/lxh_train_code/mutation/Tiles_Val_list_therapy_two_scale_norm_balance'):
            os.makedirs('Z:/admin/LXH/lxh_train_code/mutation/Tiles_Val_list_therapy_two_scale_norm_balance')
        with open('Z:/admin/LXH/lxh_train_code/mutation/Tiles_Val_list_therapy_two_scale_norm_balance/val_folder'+ str(k_fold) +'.json', 'w') as f:
            json.dump(tiles_vaild_list, f)
        with open('Z:/admin/LXH/lxh_train_code/mutation/Tiles_Val_list_therapy_two_scale_norm_balance/train_folder'+ str(k_fold) +'.json', 'w') as ft:
            json.dump(tiles_train_list, ft)

        return tiles_train_list, tiles_vaild_list



def make_test_folder(data_path, event, p_name):
        ##p_name
        
        test_p_name = p_name.values
        ##p_event /status
        
        test_p_event = event.values
        test_p_event = test_p_event.tolist()

      


       
        test_pfolders = []
        ##每一折的训练和验证的路径
        for test_pname in test_p_name:
            test_pfolder = os.path.join(data_path,test_pname)
            test_pfolders.append(test_pfolder)
        

        test_folder = [test_pfolders,test_p_event]
        

        if not os.path.exists('D:/lxh/python_learn/lxh_surv_code/mutation/Val_list_therapy_40x_norm_balance'):
            os.makedirs('D:/lxh/python_learn/lxh_surv_code/mutation/Val_list_therapy_40x_norm_balance')
        with open('D:/lxh/python_learn/lxh_surv_code/mutation/Val_list_therapy_40x_norm_balance/test_folder.json', 'w') as f:
            json.dump(test_folder, f)

        return test_folder



def make_two_scale_test_folder(data_path_10x,data_path_40x, event, p_name):
        ##p_name
        
        test_p_name = p_name.values
        ##p_event /status
        
        test_p_event = event.values
        test_p_event = test_p_event.tolist()

       
        test_pfolders_10x = []
        test_pfolders_40x = []

        ##每一折的训练和验证的路径
        for test_pname in test_p_name:
            test_pfolder_10x = os.path.join(data_path_10x,test_pname)
            test_pfolder_40x = os.path.join(data_path_40x,test_pname)

            test_pfolders_10x.append(test_pfolder_10x)
            test_pfolders_40x.append(test_pfolder_40x)

        test_folder = [test_pfolders_10x,test_pfolders_40x,test_p_event]
        


        if not os.path.exists('Z:/admin/LXH/lxh_train_code/mutation/Val_list_surgery_two_Magnify_norm_balance'):
            os.makedirs('Z:/admin/LXH/lxh_train_code/Val_list_surgery_two_Magnify_norm_balance')
        with open('Z:/admin/LXH/lxh_train_code/mutation/Val_list_surgery_two_Magnify_norm_balance/test_folder.json', 'w') as f:
            json.dump(test_folder, f)

        return test_folder



def make_tiles_two_scale_test_folder(data_path_10x,data_path_40x, event, p_name, bag_size,ratio = 1):
        ##p_name
        
        test_p_name = p_name.values
        ##p_event /status
        
        test_p_event = event.values
        test_p_event = test_p_event.tolist()

       
        test_pfolders_10x = []
        test_pfolders_40x = []

        ##每一折的训练和验证的路径
        for test_pname in test_p_name:
            test_pfolder_10x = os.path.join(data_path_10x,test_pname)
            test_pfolder_40x = os.path.join(data_path_40x,test_pname)

            test_pfolders_10x.append(test_pfolder_10x)
            test_pfolders_40x.append(test_pfolder_40x)

        test_folder = [test_pfolders_10x,test_pfolders_40x,test_p_event]
        
         ##[[10x][40x][event]]
        Tiles_folder_10x = []
        Tiles_folder_40x = []
        Tiles_event = []
        ##10x
        for p_idx,p_path in enumerate(test_folder[0]):
            
            p_event = test_folder[2][p_idx]
            total_list = glob.glob(p_path+"/*.jpg")
            total_list.sort(key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
           

            for tile in total_list:
                # tile_chunk = [tile,p_event]
                Tiles_folder_10x.append(tile)
                Tiles_event.append(p_event)

        


        ###40x
        for p_idx,p_path in enumerate(test_folder[1]):
            
            # p_event = file_dirs[2][p_idx]
            total_list = glob.glob(p_path+"/*.jpg")
            total_list.sort(key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
           

            for tile in total_list:
                # tile_chunk = [tile,p_event]
                Tiles_folder_40x.append(tile)
        # Tiles_folder = list(itertools.chain.from_iterable(Tiles_folder))
        
        list_chunks_40x = [Tiles_folder_40x[x:x+bag_size] for x in range(0, len(Tiles_folder_40x)-bag_size, bag_size)]
        list_chunks_10x = Tiles_folder_10x
        list_Tiles_event = Tiles_event

        
        

        list_chunks_10x = list_chunks_10x[:int(len(list_chunks_10x)*ratio)]
        list_chunks_40x = list_chunks_40x[:int(len(list_chunks_40x)*ratio)]

        # random.shuffle(self.list_chunks)


        test_all_list = [] 

        for idx_bag,bag_tile in  enumerate(list_chunks_40x) :
            ##[[16*tile],tile,]
            tile_10_40 = [bag_tile , list_chunks_10x[idx_bag], list_Tiles_event[idx_bag]]
            test_all_list.append(tile_10_40)

        

        
        test_all_list = test_all_list[:int(len(test_all_list)*ratio)]
        
        # print( len(self.train_all_list))

        if not os.path.exists('Z:/admin/LXH/lxh_train_code/mutation/Tiles_Val_list_therapy_two_scale_norm_balance'):
            os.makedirs('Z:/admin/LXH/lxh_train_code/Tiles_Val_list_therapy_two_scale_norm_balance')
        with open('Z:/admin/LXH/lxh_train_code/mutation/Tiles_Val_list_therapy_two_scale_norm_balance/test_folder.json', 'w') as f:
            json.dump(test_all_list, f)

        return test_folder


def pinyin(word):
    s = ''
    for i in pypinyin.pinyin(word, style=pypinyin.NORMAL):
        s += ''.join(i)
    return s


def convert20to40(data_dir, data_output_dir):
    for type_ in ('BRCA','WILD'):
        if not os.path.exists(os.path.join(data_output_dir,type_)):
            os.mkdir(os.path.join(data_output_dir,type_))
        slide_dirs = os.listdir(os.path.join(data_dir,type_))
        for slide_dir in tqdm(slide_dirs):
            slide_output_dir = os.path.join(data_output_dir,type_, slide_dir)
            if not os.path.exists(slide_output_dir):
                os.mkdir(slide_output_dir)
            jpg_dirs = glob.glob(os.path.join(data_dir,type_,slide_dir)+'/*.jpg')
            for jpg_dir in jpg_dirs:
                image_20 = Image.open(jpg_dir)
                jpg_basename = os.path.basename(jpg_dir)
                image_ul = image_20.crop((0,0,256,256))
                image_ur = image_20.crop((256,0,512,256))
                image_ll = image_20.crop((0,256,256,512))
                image_lr = image_20.crop((256,256,512,512))
                ul_name = os.path.join(slide_output_dir,jpg_basename.replace('.jpg','_ul.jpg'))
                image_ul.save(ul_name)
                ur_name = os.path.join(slide_output_dir,jpg_basename.replace('.jpg','_ur.jpg'))
                image_ur.save(ur_name)
                ll_name = os.path.join(slide_output_dir,jpg_basename.replace('.jpg','_ll.jpg'))
                image_ll.save(ll_name)
                lr_name = os.path.join(slide_output_dir,jpg_basename.replace('.jpg','_lr.jpg'))
                image_lr.save(lr_name)

def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)