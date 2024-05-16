import numpy as np
from torch.utils.data import Dataset, DataLoader
import os, glob, torch, random, itertools
from PIL import Image
import re
import random
# random.seed(1)

# class TrainDataset_Fold_Bag(Dataset):#5折用dataloader
#     def __init__(self, file_dirs, transform, bag_size, ratio = 1):
#         wild_list_=[]
#         for file_dir in file_dirs[0]:
#             pic_list = glob.glob(file_dir+"/*.jpg")
#             random.shuffle(pic_list)
#             wild_list_.append(pic_list)
#         wild_list = list(itertools.chain.from_iterable(wild_list_))
#         brca_list_=[]
#         for file_dir in file_dirs[1]:
#             pic_list = glob.glob(file_dir+"/*.jpg")
#             random.shuffle(pic_list)
#             brca_list_.append(pic_list)
#         brca_list = list(itertools.chain.from_iterable(brca_list_))
#         # random.shuffle(wild_list)
#         # random.shuffle(brca_list)
#         wild_list_chunks = [wild_list[x:x+bag_size] for x in range(0, len(wild_list)-bag_size, bag_size)]
#         brca_list_chunks = [brca_list[x:x+bag_size] for x in range(0, len(brca_list)-bag_size, bag_size)]
#         self.list_chunks = brca_list_chunks + wild_list_chunks
#         # random.shuffle(self.list_chunks)
#         self.transform = transform
#         self.bag_size = bag_size
#         self.list_chunks = self.list_chunks[:int(len(self.list_chunks)*ratio)]
#         # print('length',len(self.list_chunks))

#     def __len__(self):
#         return len(self.list_chunks)

#     def __getitem__(self, idx):
#         file_chunk = self.list_chunks[idx]
#         bag_image = torch.zeros(self.bag_size, 3, 256, 256)
#         wsi_type = 1 if 'BRCA' in file_chunk[0] else 0
#         # print(len(file_chunk))
#         for num in range(len(self.list_chunks[idx])):
#             file_name = self.list_chunks[idx][num]
#             # print(file_name)
#             image = Image.open(file_name)
#             image = self.transform(image)
#             bag_image[num,:,:,:] = image
#         return bag_image, wsi_type

# class TestDataset(Dataset):
#     def __init__(self, wsi_dir, transform, bag_size):
#         self.wsi_dir = wsi_dir
#         self.bag_size = bag_size
#         pic_list = glob.glob(self.wsi_dir+'/*.jpg')
#         # random.shuffle(pic_list)
#         self.list_chunks = [pic_list[x:x+bag_size] for x in range(0, len(pic_list)-bag_size, bag_size)]
#         # random.shuffle(self.list_chunks)
#         self.transform = transform

#     def __len__(self):
#         return len(self.list_chunks)

#     def __getitem__(self, idx):
#         bag_image = torch.zeros(self.bag_size, 3, 256, 256)
#         for num in range(self.bag_size):
#             file_name = self.list_chunks[idx][num]
#             image = Image.open(file_name)
#             image = self.transform(image)
#             bag_image[num,:,:,:] = image
#         return bag_image

# class ValDataset(Dataset):
#     def __init__(self, file_dirs, transform, bag_size):
#         wild_list_=[]
#         for file_dir in file_dirs[0]:
#             wild_list_.append(glob.glob(file_dir+"/*.jpg"))
#         wild_list = list(itertools.chain.from_iterable(wild_list_))
#         brca_list_=[]
#         for file_dir in file_dirs[1]:
#             brca_list_.append(glob.glob(file_dir+"/*.jpg"))
#         brca_list = list(itertools.chain.from_iterable(brca_list_))
#         random.shuffle(wild_list)
#         random.shuffle(brca_list)
#         wild_list_chunks = [wild_list[x:x+bag_size] for x in range(0, len(wild_list)-bag_size, bag_size)]
#         brca_list_chunks = [brca_list[x:x+bag_size] for x in range(0, len(brca_list)-bag_size, bag_size)]
#         self.list_chunks = wild_list_chunks + brca_list_chunks
#         random.shuffle(self.list_chunks)
#         self.transform = transform
#         self.bag_size = bag_size
#         print('length',len(self.list_chunks))

#     def __len__(self):
#         return len(self.list_chunks)

#     def __getitem__(self, idx):
#         file_chunk = self.list_chunks[idx]
#         bag_image = torch.zeros(self.bag_size, 3, 256, 256)
#         wsi_type = 1 if 'BRCA' in file_chunk[0] else 0
#         # print(len(file_chunk))
#         for num in range(len(self.list_chunks[idx])):
#             file_name = self.list_chunks[idx][num]
#             # print(file_name)
#             image = Image.open(file_name)
#             image = self.transform(image)
#             bag_image[num,:,:,:] = image
#         return bag_image, wsi_type
    






class TrainDataset_Fold(Dataset):#5折用dataloader
    def __init__(self, file_dirs,transform, ratio = 1):

        ###file_dirs:t_pfolders, train_p_time,train_p_event
        ###tile_folders
        Tiles_folder = []
        for p_idx,p_path in enumerate(file_dirs[0]):
            
            p_event = file_dirs[1][p_idx]
            total_list = glob.glob(p_path+"/*.jpg")
            if p_event == 0 and len(total_list) >= 1300:
                pic_list = random.sample(total_list,k = 1300)
            else:
                pic_list = total_list
            # tile_num = len(pic_list)

            for tile in pic_list:
                tile_chunk = [tile,p_event]
                Tiles_folder.append(tile_chunk)
        # Tiles_folder = list(itertools.chain.from_iterable(Tiles_folder))

        self.list_chunks = Tiles_folder
        self.transform = transform
        
        self.list_chunks = self.list_chunks[:int(len(self.list_chunks)*ratio)]
        random.shuffle(self.list_chunks)

        

    def __len__(self):
        return len(self.list_chunks)

    def __getitem__(self, idx):
        file_chunk = self.list_chunks[idx]
        image = torch.zeros(1, 3, 256, 256)

        file_name = file_chunk[0]
        label_event = file_chunk[1]
        
        image_tile = Image.open(file_name)
        image_ = self.transform(image_tile)
        image[:,:,:,:] = image_

        return  image,label_event
    



class TrainDataset_Fold_twoscale_innov(Dataset):
    def __init__(self, file_dirs, transform_original,transform_down, bag_size, ratio = 1):
        ##[[10x][40x][event]]
        Tiles_folder_10x = []
        Tiles_folder_40x = []
        Tiles_event = []
        ##10x
        for p_idx,p_path in enumerate(file_dirs[0]):
            
            p_event = file_dirs[2][p_idx]
            total_list = glob.glob(p_path+"/*.jpg")
            total_list.sort(key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
           

            for tile in total_list:
                # tile_chunk = [tile,p_event]
                Tiles_folder_10x.append(tile)
                Tiles_event.append(p_event)

        # Tiles_folder = list(itertools.chain.from_iterable(Tiles_folder))
        self.list_chunks_10x = Tiles_folder_10x
        self.list_Tiles_event = Tiles_event



        ###40x
        for p_idx,p_path in enumerate(file_dirs[1]):
            
            # p_event = file_dirs[2][p_idx]
            total_list = glob.glob(p_path+"/*.jpg")
            total_list.sort(key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
           

            for tile in total_list:
                # tile_chunk = [tile,p_event]
                Tiles_folder_40x.append(tile)
        # Tiles_folder = list(itertools.chain.from_iterable(Tiles_folder))
        self.bag_size = bag_size

        self.list_chunks_40x = [Tiles_folder_40x[x:x+bag_size] for x in range(0, len(Tiles_folder_40x)-bag_size, bag_size)]


        self.transform_original = transform_original
        self.transform_down = transform_down
        

        self.list_chunks_10x = self.list_chunks_10x[:int(len(self.list_chunks_10x)*ratio)]
        self.list_chunks_40x = self.list_chunks_40x[:int(len(self.list_chunks_40x)*ratio)]

        # random.shuffle(self.list_chunks)


        train_all_list = [] 

        for idx_bag,bag_tile in  enumerate(self.list_chunks_40x) :
            ##[[16*tile],tile,]
            tile_10_40 = [bag_tile , self.list_chunks_10x[idx_bag], self.list_Tiles_event[idx_bag]]
            train_all_list.append(tile_10_40)

        

        self.train_all_list = train_all_list
        self.train_all_list = self.train_all_list[:int(len(self.train_all_list)*ratio)]
        ##平衡数据集
        list_chunk_0 = []
        list_chunk_1 = []

        for idx_chunk,i_chunk  in enumerate(self.train_all_list):
            if i_chunk[2] == 0:
                list_chunk_0.append(i_chunk)

            else:
                list_chunk_1.append(i_chunk)
        
        list_chunk_0_select = random.sample(list_chunk_0,k = int(len(list_chunk_1)*1.1))

        
        train_all_list_select = [list_chunk_0_select,list_chunk_1]
        train_all_list_select = list(itertools.chain.from_iterable(train_all_list_select))

        self.train_all_list_select = train_all_list_select
        self.train_all_list_select = self.train_all_list_select[:int(len(self.train_all_list_select)*ratio)]


        random.shuffle(self.train_all_list_select)
        # print( len(self.train_all_list))


    def __len__(self):
        return  len(self.train_all_list_select)
        # return  len(self.list_40x_all)



    def __getitem__(self,idx):
        ##40x
        ##[40,10]
        
        bag_image_40x = torch.zeros(self.bag_size, 3, 256, 256)
        # bag_image_down_40x = torch.zeros(self.bag_size, 3, 128, 128)

        for num in range(self.bag_size):
            file_name_40x = self.train_all_list_select[idx][0][num]
            # print(file_name_40x)
            image_40x = Image.open(file_name_40x)

            image_40x_ = self.transform_original(image_40x)
            # image_down_40x = self.transform_down(image_40x)


            bag_image_40x[num,:,:,:] = image_40x_
            # bag_image_down_40x[num,:,:,:] = image_down_40x
        
                

        ##10x
        title_image_10x = torch.zeros(1, 3, 256, 256)
        title_image_10x_down = torch.zeros(1, 3, 128, 128)

        file_name_10x = self.train_all_list_select[idx][1] 
        wsi_type = self.train_all_list_select[idx][2]   
        image_10x = Image.open(file_name_10x)
        image_10x_original = self.transform_original(image_10x)
        image_10x_down = self.transform_down(image_10x)


        title_image_10x[:,:,:,:] = image_10x_original
        title_image_10x_down[:,:,:,:] = image_10x_down



        # print(file_name_10x)
        return bag_image_40x,title_image_10x,title_image_10x_down,wsi_type






class TrainDataset_Fold_twoscale_sliderandom_innov(Dataset):
    def __init__(self, file_dirs, transform_original,transform_down, bag_size, ratio = 1):
        ##[[10x][40x][event]]
        Tiles_folder_10x = []
        Tiles_folder_40x = []
        Tiles_event = []
        ##10x
        self.bag_size = bag_size

        for p_idx,(p_path_10x,p_path_40x) in enumerate (zip(file_dirs[0],file_dirs[1])):

        # for p_idx,p_path in enumerate(file_dirs[0]):
            
            p_event = file_dirs[2][p_idx]

            ##10x
            total_list_10x = glob.glob(p_path_10x+"/*.jpg")
            total_list_10x.sort(key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

            ##40x
            total_list_40x = glob.glob(p_path_40x+"/*.jpg")
            total_list_40x.sort(key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
           

            random.seed(0)

            ##随机挑选100个，获得index
            if p_event == 0 and len(total_list_10x) > 100:
                index_pic_chunk = random.sample(list(enumerate(total_list_10x)),100)
                select_index = []
                pic_list_10x = []
                for idx ,pic in index_pic_chunk:
                    select_index.append(idx)
                    pic_list_10x.append(pic)

              

            else:
                select_index = range(len(total_list_10x)-1)
                pic_list_10x = total_list_10x

            for tile in pic_list_10x:
                # tile_chunk = [tile,p_event]
                Tiles_folder_10x.append(tile)
                Tiles_event.append(p_event)

            ###每张side按照bag来产生list
            slide_tile_folder_40x = []    
            for tile in total_list_40x:
                # tile_chunk = [tile,p_event]
                slide_tile_folder_40x.append(tile)
            slide_list_chunks_40x = [slide_tile_folder_40x[x:x+bag_size] for x in range(0, len(slide_tile_folder_40x)-bag_size +1 , bag_size)]


            for i_bag in select_index:
                Tiles_folder_40x.append(slide_list_chunks_40x[i_bag])
            

        # Tiles_folder = list(itertools.chain.from_iterable(Tiles_folder))
        self.list_chunks_10x = Tiles_folder_10x
        self.list_Tiles_event = Tiles_event
        self.list_chunks_40x = Tiles_folder_40x


        self.transform_original = transform_original
        self.transform_down = transform_down
        

        self.list_chunks_10x = self.list_chunks_10x[:int(len(self.list_chunks_10x)*ratio)]
        self.list_chunks_40x = self.list_chunks_40x[:int(len(self.list_chunks_40x)*ratio)]

        # random.shuffle(self.list_chunks)


        train_all_list = [] 

        for idx_bag,bag_tile in  enumerate(self.list_chunks_40x) :
            ##[[16*tile],tile,]
            tile_10_40 = [bag_tile , self.list_chunks_10x[idx_bag], self.list_Tiles_event[idx_bag]]
            train_all_list.append(tile_10_40)

        

        self.train_all_list = train_all_list

        self.train_all_list = self.train_all_list[:int(len(self.train_all_list)*ratio)]
        random.shuffle(self.train_all_list)

       


    def __len__(self):
        return  len(self.train_all_list)
        # return  len(self.list_40x_all)



    def __getitem__(self,idx):
        ##40x
        ##[40,10]
        
        bag_image_40x = torch.zeros(self.bag_size, 3, 256, 256)
        # bag_image_down_40x = torch.zeros(self.bag_size, 3, 128, 128)

        for num in range(self.bag_size):
            file_name_40x = self.train_all_list[idx][0][num]
            # print(file_name_40x)
            image_40x = Image.open(file_name_40x)

            image_40x_ = self.transform_original(image_40x)
            # image_down_40x = self.transform_down(image_40x)


            bag_image_40x[num,:,:,:] = image_40x_
            # bag_image_down_40x[num,:,:,:] = image_down_40x
        
                

        ##10x
        title_image_10x = torch.zeros(1, 3, 256, 256)
        title_image_10x_down = torch.zeros(1, 3, 128, 128)

        file_name_10x = self.train_all_list[idx][1] 
        wsi_type = self.train_all_list[idx][2]   
        image_10x = Image.open(file_name_10x)
        image_10x_original = self.transform_original(image_10x)
        image_10x_down = self.transform_down(image_10x)


        title_image_10x[:,:,:,:] = image_10x_original
        title_image_10x_down[:,:,:,:] = image_10x_down



        # print(file_name_10x)
        return bag_image_40x,title_image_10x,title_image_10x_down,wsi_type





class TrainDataset_Fold_two_Magnify_sliderandom_innov(Dataset):
    def __init__(self, file_dirs, transform_original,transform_down, ratio = 1):
        ##[[10x][40x][event]]
        Tiles_folder_10x = []
        # Tiles_folder_40x = []
        Tiles_event = []
        ##10x
        

        for p_idx,p_path_10x in enumerate (file_dirs[0]):

        # for p_idx,p_path in enumerate(file_dirs[0]):
            
            p_event = file_dirs[2][p_idx]

            ##10x
            total_list_10x = glob.glob(p_path_10x+"/*.jpg")
            total_list_10x.sort(key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

           

            random.seed(0)

            ##随机挑选100个，获得index
            if p_event == 0 and len(total_list_10x) > 90:
                index_pic_chunk = random.sample(list(enumerate(total_list_10x)),90)
                select_index = []
                pic_list_10x = []
                for idx ,pic in index_pic_chunk:
                    select_index.append(idx)
                    pic_list_10x.append(pic)

              

            else:
                select_index = range(len(total_list_10x)-1)
                pic_list_10x = total_list_10x

            for tile in pic_list_10x:
                # tile_chunk = [tile,p_event]
                Tiles_folder_10x.append(tile)
                Tiles_event.append(p_event)

            
        # Tiles_folder = list(itertools.chain.from_iterable(Tiles_folder))
        self.list_chunks_10x = Tiles_folder_10x
        self.list_Tiles_event = Tiles_event
        

        self.transform_original = transform_original
        self.transform_down = transform_down
        

        self.list_chunks_10x = self.list_chunks_10x[:int(len(self.list_chunks_10x)*ratio)]
        

        # random.shuffle(self.list_chunks)


        train_all_list = [] 

        for idx_bag,tile_10x in  enumerate(self.list_chunks_10x) :
            ##[[16*tile],tile,]
            tile_10_10 = [tile_10x, self.list_Tiles_event[idx_bag]]
            train_all_list.append(tile_10_10)

        

        self.train_all_list = train_all_list

        self.train_all_list = self.train_all_list[:int(len(self.train_all_list)*ratio)]
        random.shuffle(self.train_all_list)

       


    def __len__(self):
        return  len(self.train_all_list)
        # return  len(self.list_40x_all)



    def __getitem__(self,idx):
          

        ##10x
        title_image_10x = torch.zeros(1, 3, 256, 256)
        title_image_10x_down = torch.zeros(1, 3, 128, 128)

        file_name_10x = self.train_all_list[idx][0] 
        wsi_type = self.train_all_list[idx][1]   
        image_10x = Image.open(file_name_10x)
        image_10x_original = self.transform_original(image_10x)
        image_10x_down = self.transform_down(image_10x)


        title_image_10x[:,:,:,:] = image_10x_original
        title_image_10x_down[:,:,:,:] = image_10x_down



        # print(file_name_10x)
        return title_image_10x,title_image_10x_down,wsi_type




class TrainDataset_Fold_4010_innov(Dataset):
    def __init__(self, file_dirs, transform_original,transform_down, bag_size, ratio = 1):
        ##[[10x][40x][event]]
        Tiles_folder_10x = []
        Tiles_folder_40x = []
        Tiles_event = []
        ##10x
        for p_idx,p_path in enumerate(file_dirs[0]):
            
            p_event = file_dirs[2][p_idx]
            total_list = glob.glob(p_path+"/*.jpg")
            total_list.sort(key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
           

            for tile in total_list:
                # tile_chunk = [tile,p_event]
                Tiles_folder_10x.append(tile)
                Tiles_event.append(p_event)

        # Tiles_folder = list(itertools.chain.from_iterable(Tiles_folder))
        self.list_chunks_10x = Tiles_folder_10x
        self.list_Tiles_event = Tiles_event



        ###40x
        for p_idx,p_path in enumerate(file_dirs[1]):
            
            # p_event = file_dirs[2][p_idx]
            total_list = glob.glob(p_path+"/*.jpg")
            total_list.sort(key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
           

            for tile in total_list:
                # tile_chunk = [tile,p_event]
                Tiles_folder_40x.append(tile)
        # Tiles_folder = list(itertools.chain.from_iterable(Tiles_folder))
        self.bag_size = bag_size

        self.list_chunks_40x = [Tiles_folder_40x[x:x+bag_size] for x in range(0, len(Tiles_folder_40x)-bag_size, bag_size)]


        self.transform_original = transform_original
        self.transform_down = transform_down
        

        self.list_chunks_10x = self.list_chunks_10x[:int(len(self.list_chunks_10x)*ratio)]
        self.list_chunks_40x = self.list_chunks_40x[:int(len(self.list_chunks_40x)*ratio)]

        # random.shuffle(self.list_chunks)


        train_all_list = [] 

        for idx_bag,bag_tile in  enumerate(self.list_chunks_40x) :
            ##[[16*tile],tile,]
            tile_10_40 = [bag_tile , self.list_chunks_10x[idx_bag], self.list_Tiles_event[idx_bag]]
            train_all_list.append(tile_10_40)

        

        self.train_all_list = train_all_list
        self.train_all_list = self.train_all_list[:int(len(self.train_all_list)*ratio)]
        ##平衡数据集
        list_chunk_0 = []
        list_chunk_1 = []

        for idx_chunk,i_chunk  in enumerate(self.train_all_list):
            if i_chunk[2] == 0:
                list_chunk_0.append(i_chunk)

            else:
                list_chunk_1.append(i_chunk)
        
        list_chunk_0_select = random.sample(list_chunk_0,k = int(len(list_chunk_1)*1.5))

        
        train_all_list_select = [list_chunk_0_select,list_chunk_1]
        train_all_list_select = list(itertools.chain.from_iterable(train_all_list_select))

        self.train_all_list_select = train_all_list_select
        self.train_all_list_select = self.train_all_list_select[:int(len(self.train_all_list_select)*ratio)]


        random.shuffle(self.train_all_list_select)
        # print( len(self.train_all_list))


    def __len__(self):
        return  len(self.train_all_list_select)
        # return  len(self.list_40x_all)



    def __getitem__(self,idx):
        ##40x
        ##[40,10]
        
        bag_image_40x = torch.zeros(self.bag_size, 3, 256, 256)
        # bag_image_down_40x = torch.zeros(self.bag_size, 3, 128, 128)

        for num in range(self.bag_size):
            file_name_40x = self.train_all_list_select[idx][0][num]
            # print(file_name_40x)
            image_40x = Image.open(file_name_40x)

            image_40x_ = self.transform_original(image_40x)
            # image_down_40x = self.transform_down(image_40x)


            bag_image_40x[num,:,:,:] = image_40x_
            # bag_image_down_40x[num,:,:,:] = image_down_40x
        
                

        ##10x
        title_image_10x = torch.zeros(1, 3, 256, 256)
       

        file_name_10x = self.train_all_list_select[idx][1] 
        wsi_type = self.train_all_list_select[idx][2]   
        image_10x = Image.open(file_name_10x)
        image_10x_original = self.transform_original(image_10x)
        # image_10x_down = self.transform_down(image_10x)


        title_image_10x[:,:,:,:] = image_10x_original
        



        # print(file_name_10x)
        return bag_image_40x,title_image_10x,wsi_type







class TrainDataset_Fold_Tiles_twoscale_innov(Dataset):
    def __init__(self, file_dirs, transform_original,transform_down, bag_size):
       
        self.bag_size = bag_size
        self.transform_original = transform_original
        self.transform_down = transform_down
        self.train_all_list_select = file_dirs


        random.shuffle(self.train_all_list_select)
        # print( len(self.train_all_list))


    def __len__(self):
        return  len(self.train_all_list_select)
        # return  len(self.list_40x_all)



    def __getitem__(self,idx):
        ##40x
        ##[40,10]
        
        bag_image_40x = torch.zeros(self.bag_size, 3, 256, 256)
        bag_image_down_40x = torch.zeros(self.bag_size, 3, 128, 128)

        for num in range(self.bag_size):
            file_name_40x = self.train_all_list_select[idx][0][num]
            # print(file_name_40x)
            image_40x = Image.open(file_name_40x)

            image_40x_ = self.transform_original(image_40x)
            image_down_40x = self.transform_down(image_40x)


            bag_image_40x[num,:,:,:] = image_40x_
            bag_image_down_40x[num,:,:,:] = image_down_40x
        
                

        ##10x
        title_image_10x = torch.zeros(1, 3, 256, 256)
        file_name_10x = self.train_all_list_select[idx][1] 
        wsi_type = self.train_all_list_select[idx][2]   
        image_10x = Image.open(file_name_10x)
        image_10x = self.transform_original(image_10x)
        title_image_10x[:,:,:,:] = image_10x

        # print(file_name_10x)
        return bag_image_40x, bag_image_down_40x,title_image_10x,wsi_type
    



class ValDataset_Fold_twoscale_innov(Dataset):
    def __init__(self, file_dirs, transform_original,transform_down, bag_size, ratio = 1):
        ##[[10x][40x][event]]
        Tiles_folder_10x = []
        Tiles_folder_40x = []
        Tiles_event = []
        ##10x
        for p_idx,p_path in enumerate(file_dirs[0]):
            
            p_event = file_dirs[2][p_idx]
            total_list = glob.glob(p_path+"/*.jpg")
            total_list.sort(key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
           

            for tile in total_list:
                # tile_chunk = [tile,p_event]
                Tiles_folder_10x.append(tile)
                Tiles_event.append(p_event)

        # Tiles_folder = list(itertools.chain.from_iterable(Tiles_folder))
        self.list_chunks_10x = Tiles_folder_10x
        self.list_Tiles_event = Tiles_event



        ###40x
        for p_idx,p_path in enumerate(file_dirs[1]):
            
            # p_event = file_dirs[2][p_idx]
            total_list = glob.glob(p_path+"/*.jpg")
            total_list.sort(key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
           

            for tile in total_list:
                # tile_chunk = [tile,p_event]
                Tiles_folder_40x.append(tile)
        # Tiles_folder = list(itertools.chain.from_iterable(Tiles_folder))
        self.bag_size = bag_size

        self.list_chunks_40x = [Tiles_folder_40x[x:x+bag_size] for x in range(0, len(Tiles_folder_40x)-bag_size+1, bag_size)]


        self.transform_original = transform_original
        self.transform_down = transform_down
        

        self.list_chunks_10x = self.list_chunks_10x[:int(len(self.list_chunks_10x)*ratio)]
        self.list_chunks_40x = self.list_chunks_40x[:int(len(self.list_chunks_40x)*ratio)]

        # random.shuffle(self.list_chunks)


        train_all_list = [] 

        for idx_bag,bag_tile in  enumerate(self.list_chunks_40x) :
            ##[[16*tile],tile,]
            tile_10_40 = [bag_tile , self.list_chunks_10x[idx_bag],self.list_Tiles_event[idx_bag]]
            train_all_list.append(tile_10_40)

        

        self.train_all_list = train_all_list
        self.train_all_list = self.train_all_list[:int(len(self.train_all_list)*ratio)]
       


        random.shuffle(self.train_all_list)
        # print( len(self.train_all_list))


    def __len__(self):
        return  len(self.train_all_list)
        # return  len(self.list_40x_all)



    def __getitem__(self,idx):
        ##40x
        ##[40,10]
        
        bag_image_40x = torch.zeros(self.bag_size, 3, 256, 256)
        # bag_image_down_40x = torch.zeros(self.bag_size, 3, 128, 128)

        for num in range(self.bag_size):
            file_name_40x = self.train_all_list[idx][0][num]
            # print(file_name_40x)
            image_40x = Image.open(file_name_40x)

            image_40x_ = self.transform_original(image_40x)
            # image_down_40x = self.transform_down(image_40x)


            bag_image_40x[num,:,:,:] = image_40x_
            # bag_image_down_40x[num,:,:,:] = image_down_40x
        
                

        ##10x
        title_image_10x = torch.zeros(1, 3, 256, 256)
        title_image_10x_down = torch.zeros(1, 3, 128, 128)

        file_name_10x = self.train_all_list[idx][1] 
        wsi_type = self.train_all_list[idx][2]   
        image_10x = Image.open(file_name_10x)

        image_10x_original = self.transform_original(image_10x)
        image_10x_down = self.transform_down(image_10x)

        title_image_10x[:,:,:,:] = image_10x_original
        title_image_10x_down[:,:,:,:] = image_10x_down


        # print(file_name_10x)
        return bag_image_40x,title_image_10x,title_image_10x_down,wsi_type
    





class ValDataset_Fold_two_Magnify_innov(Dataset):
    def __init__(self, file_dirs, transform_original,transform_down, ratio = 1):
        ##[[10x][40x][event]]
        Tiles_folder_10x = []
        
        Tiles_event = []
        ##10x
        for p_idx,p_path in enumerate(file_dirs[0]):
            
            p_event = file_dirs[2][p_idx]
            total_list = glob.glob(p_path+"/*.jpg")
            total_list.sort(key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
           

            for tile in total_list:
                # tile_chunk = [tile,p_event]
                Tiles_folder_10x.append(tile)
                Tiles_event.append(p_event)

        # Tiles_folder = list(itertools.chain.from_iterable(Tiles_folder))
        self.list_chunks_10x = Tiles_folder_10x
        self.list_Tiles_event = Tiles_event

        

        self.transform_original = transform_original
        self.transform_down = transform_down
        

        self.list_chunks_10x = self.list_chunks_10x[:int(len(self.list_chunks_10x)*ratio)]
        

        train_all_list = [] 

        for idx_bag,tile_10x in  enumerate(self.list_chunks_10x) :
            ##[[16*tile],tile,]
            tile_10_10 = [tile_10x,self.list_Tiles_event[idx_bag]]
            train_all_list.append(tile_10_10)

        

        self.train_all_list = train_all_list
        self.train_all_list = self.train_all_list[:int(len(self.train_all_list)*ratio)]
       


        random.shuffle(self.train_all_list)
        # print( len(self.train_all_list))


    def __len__(self):
        return  len(self.train_all_list)
        # return  len(self.list_40x_all)



    def __getitem__(self,idx):
       

        ##10x
        title_image_10x = torch.zeros(1, 3, 256, 256)
        title_image_10x_down = torch.zeros(1, 3, 128, 128)

        file_name_10x = self.train_all_list[idx][0] 
        wsi_type = self.train_all_list[idx][1]   
        image_10x = Image.open(file_name_10x)

        image_10x_original = self.transform_original(image_10x)
        image_10x_down = self.transform_down(image_10x)

        title_image_10x[:,:,:,:] = image_10x_original
        title_image_10x_down[:,:,:,:] = image_10x_down


        # print(file_name_10x)
        return title_image_10x,title_image_10x_down,wsi_type
    




class ValDataset_Fold_4010_innov(Dataset):
    def __init__(self, file_dirs, transform_original,transform_down, bag_size, ratio = 1):
        ##[[10x][40x][event]]
        Tiles_folder_10x = []
        Tiles_folder_40x = []
        Tiles_event = []
        ##10x
        for p_idx,p_path in enumerate(file_dirs[0]):
            
            p_event = file_dirs[2][p_idx]
            total_list = glob.glob(p_path+"/*.jpg")
            total_list.sort(key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
           

            for tile in total_list:
                # tile_chunk = [tile,p_event]
                Tiles_folder_10x.append(tile)
                Tiles_event.append(p_event)

        # Tiles_folder = list(itertools.chain.from_iterable(Tiles_folder))
        self.list_chunks_10x = Tiles_folder_10x
        self.list_Tiles_event = Tiles_event



        ###40x
        for p_idx,p_path in enumerate(file_dirs[1]):
            
            # p_event = file_dirs[2][p_idx]
            total_list = glob.glob(p_path+"/*.jpg")
            total_list.sort(key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
           

            for tile in total_list:
                # tile_chunk = [tile,p_event]
                Tiles_folder_40x.append(tile)
        # Tiles_folder = list(itertools.chain.from_iterable(Tiles_folder))
        self.bag_size = bag_size

        self.list_chunks_40x = [Tiles_folder_40x[x:x+bag_size] for x in range(0, len(Tiles_folder_40x)-bag_size, bag_size)]


        self.transform_original = transform_original
        self.transform_down = transform_down
        

        self.list_chunks_10x = self.list_chunks_10x[:int(len(self.list_chunks_10x)*ratio)]
        self.list_chunks_40x = self.list_chunks_40x[:int(len(self.list_chunks_40x)*ratio)]

        # random.shuffle(self.list_chunks)


        train_all_list = [] 

        for idx_bag,bag_tile in  enumerate(self.list_chunks_40x) :
            ##[[16*tile],tile,]
            tile_10_40 = [bag_tile , self.list_chunks_10x[idx_bag],self.list_Tiles_event[idx_bag]]
            train_all_list.append(tile_10_40)

        

        self.train_all_list = train_all_list
        self.train_all_list = self.train_all_list[:int(len(self.train_all_list)*ratio)]
       


        random.shuffle(self.train_all_list)
        # print( len(self.train_all_list))


    def __len__(self):
        return  len(self.train_all_list)
        # return  len(self.list_40x_all)



    def __getitem__(self,idx):
        ##40x
        ##[40,10]
        
        bag_image_40x = torch.zeros(self.bag_size, 3, 256, 256)
        # bag_image_down_40x = torch.zeros(self.bag_size, 3, 128, 128)

        for num in range(self.bag_size):
            file_name_40x = self.train_all_list[idx][0][num]
            # print(file_name_40x)
            image_40x = Image.open(file_name_40x)

            image_40x_ = self.transform_original(image_40x)
            # image_down_40x = self.transform_down(image_40x)


            bag_image_40x[num,:,:,:] = image_40x_
            # bag_image_down_40x[num,:,:,:] = image_down_40x
        
                

        ##10x
        title_image_10x = torch.zeros(1, 3, 256, 256)
        # title_image_10x_down = torch.zeros(1, 3, 128, 128)

        file_name_10x = self.train_all_list[idx][1] 
        wsi_type = self.train_all_list[idx][2]   
        image_10x = Image.open(file_name_10x)

        image_10x_original = self.transform_original(image_10x)
        # image_10x_down = self.transform_down(image_10x)

        title_image_10x[:,:,:,:] = image_10x_original
        # title_image_10x_down[:,:,:,:] = image_10x_down


        # print(file_name_10x)
        return bag_image_40x,title_image_10x,wsi_type
    

    
class TestDataset_Fold_twoscale_innov(Dataset):
    def __init__(self, i_slide,file_dirs, transform_original,transform_down, bag_size, ratio = 1):
        ##[[10x][40x][event]]
        Tiles_folder_10x = []
        Tiles_folder_40x = []
        Tiles_event = []
        ##10x
        p_path_10x = file_dirs[0][i_slide]
        p_path_40x = file_dirs[1][i_slide]
        p_event = file_dirs[2][i_slide]

        total_list_10x = glob.glob(p_path_10x+"/*.jpg")
        total_list_10x.sort(key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
    
        total_list_40x = glob.glob(p_path_40x+"/*.jpg")
        total_list_40x.sort(key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
    

        for tile in total_list_10x:
            # tile_chunk = [tile,p_event]
            Tiles_folder_10x.append(tile)
            Tiles_event.append(p_event)

        for tile in total_list_40x:
            # tile_chunk = [tile,p_event]
            Tiles_folder_40x.append(tile)
        
        self.list_chunks_10x = Tiles_folder_10x
        self.list_Tiles_event = Tiles_event



        self.bag_size = bag_size

        self.list_chunks_40x = [Tiles_folder_40x[x:x+bag_size] for x in range(0, len(Tiles_folder_40x)-bag_size, bag_size)]


        self.transform_original = transform_original
        self.transform_down = transform_down
        

        self.list_chunks_10x = self.list_chunks_10x[:int(len(self.list_chunks_10x)*ratio)]
        self.list_chunks_40x = self.list_chunks_40x[:int(len(self.list_chunks_40x)*ratio)]

        # random.shuffle(self.list_chunks)


        train_all_list = [] 

        for idx_bag,bag_tile in  enumerate(self.list_chunks_40x) :
            ##[[16*tile],tile,]
            tile_10_40 = [bag_tile , self.list_chunks_10x[idx_bag], self.list_Tiles_event[idx_bag]]
            train_all_list.append(tile_10_40)

        

        self.train_all_list = train_all_list
        self.train_all_list = self.train_all_list[:int(len(self.train_all_list)*ratio)]
       


        random.shuffle(self.train_all_list)
        # print( len(self.train_all_list))


    def __len__(self):
        return  len(self.train_all_list)
        # return  len(self.list_40x_all)



    def __getitem__(self,idx):
        ##40x
        ##[40,10]
        
        bag_image_40x = torch.zeros(self.bag_size, 3, 256, 256)
        # bag_image_down_40x = torch.zeros(self.bag_size, 3, 128, 128)

        for num in range(self.bag_size):
            file_name_40x = self.train_all_list[idx][0][num]
            # print(file_name_40x)
            image_40x = Image.open(file_name_40x)

            image_40x_ = self.transform_original(image_40x)
            # image_down_40x = self.transform_down(image_40x)


            bag_image_40x[num,:,:,:] = image_40x_
            # bag_image_down_40x[num,:,:,:] = image_down_40x
        
                

        ##10x
        title_image_10x = torch.zeros(1, 3, 256, 256)
        title_image_10x_down = torch.zeros(1, 3, 128, 128)

        file_name_10x = self.train_all_list[idx][1] 
        wsi_type = self.train_all_list[idx][2]   
        image_10x = Image.open(file_name_10x)
        image_10x_original = self.transform_original(image_10x)
        image_10x_down = self.transform_down(image_10x)
        title_image_10x[:,:,:,:] = image_10x_original
        title_image_10x_down[:,:,:,:] = image_10x_down


        # print(file_name_10x)
        return bag_image_40x,title_image_10x,title_image_10x_down,wsi_type
    


class TestDataset_Fold_two_Magnify_innov(Dataset):
    def __init__(self, i_slide,file_dirs, transform_original,transform_down, ratio = 1):
        ##[[10x][40x][event]]
        Tiles_folder_10x = []
        Tiles_event = []
        ##10x
        p_path_10x = file_dirs[0][i_slide]
        p_event = file_dirs[2][i_slide]

        total_list_10x = glob.glob(p_path_10x+"/*.jpg")
        total_list_10x.sort(key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
    
        
        for tile in total_list_10x:
            # tile_chunk = [tile,p_event]
            Tiles_folder_10x.append(tile)
            Tiles_event.append(p_event)

   
        
        self.list_chunks_10x = Tiles_folder_10x
        self.list_Tiles_event = Tiles_event



        self.transform_original = transform_original
        self.transform_down = transform_down
        

        self.list_chunks_10x = self.list_chunks_10x[:int(len(self.list_chunks_10x)*ratio)]
 

        # random.shuffle(self.list_chunks)


        train_all_list = [] 

        for idx_bag,bag_tile in  enumerate(self.list_chunks_10x) :
            ##[[16*tile],tile,]
            tile_10_10 = [bag_tile , self.list_Tiles_event[idx_bag]]
            train_all_list.append(tile_10_10)

        

        self.train_all_list = train_all_list
        self.train_all_list = self.train_all_list[:int(len(self.train_all_list)*ratio)]

        random.shuffle(self.train_all_list)
        # print( len(self.train_all_list))


    def __len__(self):
        return  len(self.train_all_list)
        # return  len(self.list_40x_all)



    def __getitem__(self,idx):
    
        ##10x
        title_image_10x = torch.zeros(1, 3, 256, 256)
        title_image_10x_down = torch.zeros(1, 3, 128, 128)

        file_name_10x = self.train_all_list[idx][0] 
        wsi_type = self.train_all_list[idx][1]   
        image_10x = Image.open(file_name_10x)
        image_10x_original = self.transform_original(image_10x)
        image_10x_down = self.transform_down(image_10x)
        title_image_10x[:,:,:,:] = image_10x_original
        title_image_10x_down[:,:,:,:] = image_10x_down


        # print(file_name_10x)
        return title_image_10x,title_image_10x_down,wsi_type
    


class ValDataset_Fold(Dataset):#5折用dataloader
    def __init__(self, file_dirs,transform, ratio = 1):

        ###file_dirs:t_pfolders, train_p_time,train_p_event
        ###tile_folders
        Tiles_folder = []
        for p_idx,p_path in enumerate(file_dirs[0]):
            
            p_event = file_dirs[1][p_idx]
            total_list = glob.glob(p_path+"/*.jpg")
          
            for tile in total_list:
                tile_chunk = [tile,p_event]
                Tiles_folder.append(tile_chunk)
        # Tiles_folder = list(itertools.chain.from_iterable(Tiles_folder))

        self.list_chunks = Tiles_folder
        self.transform = transform
        
        self.list_chunks = self.list_chunks[:int(len(self.list_chunks)*ratio)]
        random.shuffle(self.list_chunks)

        

    def __len__(self):
        return len(self.list_chunks)

    def __getitem__(self, idx):
        file_chunk = self.list_chunks[idx]
        image = torch.zeros(1, 3, 256, 256)

        file_name = file_chunk[0]
        label_event = file_chunk[1]
        
        image_tile = Image.open(file_name)
        image_ = self.transform(image_tile)
        image[:,:,:,:] = image_
        
        return  image,label_event
    




class Test_slide_Dataset_Fold(Dataset):#5折用dataloader
    def __init__(self,i_slide, file_dirs,transform, ratio = 1):

        ###file_dirs:t_pfolders, train_p_time,train_p_event
        ###tile_folders
        Tiles_folder = []
        p_path = file_dirs[0][i_slide]
            
        p_event = file_dirs[1][i_slide]
        total_list = glob.glob(p_path+"/*.jpg")
        
        for tile in total_list:
            tile_chunk = [tile,p_event]
            Tiles_folder.append(tile_chunk)
        # Tiles_folder = list(itertools.chain.from_iterable(Tiles_folder))

        self.list_chunks = Tiles_folder
        self.transform = transform
        
        self.list_chunks = self.list_chunks[:int(len(self.list_chunks)*ratio)]
        random.shuffle(self.list_chunks)

        

    def __len__(self):
        return len(self.list_chunks)

    def __getitem__(self, idx):
        file_chunk = self.list_chunks[idx]
        image = torch.zeros(1, 3, 224, 224)

        file_name = file_chunk[0]
        label_event = file_chunk[1]
        
        image_tile = Image.open(file_name)
        image_ = self.transform(image_tile)
        image[:,:,:,:] = image_
        
        return  image,label_event
    

class Test_tiles_Dataset_Fold(Dataset):#5折用dataloader
    def __init__(self, file_dirs,transform, ratio = 1):

        ###file_dirs:t_pfolders, train_p_time,train_p_event
        ###tile_folders
        Tiles_folder = []
        for p_idx,p_path in enumerate(file_dirs[0]):
            
            p_event = file_dirs[1][p_idx]
            total_list = glob.glob(p_path+"/*.jpg")
          
            for tile in total_list:
                tile_chunk = [tile,p_event]
                Tiles_folder.append(tile_chunk)
        # Tiles_folder = list(itertools.chain.from_iterable(Tiles_folder))

        self.list_chunks = Tiles_folder
        self.transform = transform
        
        self.list_chunks = self.list_chunks[:int(len(self.list_chunks)*ratio)]
        random.shuffle(self.list_chunks)

        

    def __len__(self):
        return len(self.list_chunks)

    def __getitem__(self, idx):
        file_chunk = self.list_chunks[idx]
        image = torch.zeros(1, 3, 224, 224)

        file_name = file_chunk[0]
        label_event = file_chunk[1]
        
        image_tile = Image.open(file_name)
        image_ = self.transform(image_tile)
        image[:,:,:,:] = image_
        
        return  file_name,image,label_event
    