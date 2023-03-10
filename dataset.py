import numpy as np
import cv2
from torch.utils.data import Dataset 
import os
import random

# from collections import OrderDict


def transform(image,size=(512,512)):
    """
    Arguments:
        image(np.array)：shape may be (H,W,3) or (H,W)
    Return:
        resized image or mask
    """
    channel_num = len(image.shape)
    if channel_num == 2:
        image = cv2.resize(image,size,interpolation=cv2.INTER_NEAREST)/255.0
        image = image.astype(np.float32)

    else:
        image = cv2.resize(image,size,interpolation=cv2.INTER_LINEAR)
        image = image/255.0
        image = image.transpose(2,0,1).astype(np.float32)
        
    return image


class SarShipTrain(Dataset):

    def __init__(self,root_path, data_class, transform=transform,mode='train'):

        self.root_path = root_path
        self.mode = mode
        self.data_classes = data_class    #options:['QD','IS','HK','SH']

        print('{} on {}'.format(self.mode,  self.data_classes))
        self.group_images = self.get_group_image()  # get all areas image path
        self.transform = transform

        self.mapMatrix = self.mapcolor()

    def mapcolor(self):
        '''
        read images by opencv:the channel order : [B,G,R]
        '''
        colormap = [[240, 241, 242],
                    [255, 218, 170],
                    [174, 136, 192]]

        mapMatrix = np.zeros(256 * 256 * 256, dtype=np.int32)
        for i, cm in enumerate(colormap):
            mapMatrix[cm[0] * 65536 + cm[1] * 256 + cm[2]] = i

        return mapMatrix

    def color2digit(self, label, mapMatrix):

        data = label.astype('int32')
        index = data[:, :, 0] * 65536 + data[:, :, 1] * 256 + data[:, :, 2]
        digital = mapMatrix[index]

        return digital

    def get_group_image(self):

        group_images = dict()
        for data_class in self.data_classes:
            group_images[data_class] = []
            for data in os.listdir(os.path.join(self.root_path,data_class,'data')):
                item = {}
                item['data'] = os.path.join(os.path.join(self.root_path,data_class,'data',data))
                item['mask'] = os.path.join(os.path.join(self.root_path,data_class,'mask',data))
                group_images[data_class].append(item)
                
        return group_images

    def __len__(self):

        return 666666666

    def __getitem__(self,index):

        rand_class = np.random.choice(self.data_classes,1,replace=False)
        pair_images = np.random.choice(self.group_images[rand_class[0]],2,replace=False)
        
        # read image in three channels
        support_img = cv2.imread(pair_images[0]['data'],cv2.IMREAD_COLOR)
        support_img = self.transform(support_img)

        # read mask in single channels
        support_img_mask = cv2.imread(pair_images[0]['mask'],-1)
        if len(support_img_mask.shape) == 3:
            support_img_mask = self.color2digit(support_img_mask,self.mapMatrix)
            support_img_mask = ((support_img_mask==2)*1).astype(np.float32)
        else:
            support_img_mask = self.transform(support_img_mask)           
        # print('==suport',np.unique(support_img_mask),pair_images[0]['mask'])

        query_img = cv2.imread(pair_images[1]['data'],cv2.IMREAD_COLOR)
        query_img = self.transform(query_img)

        query_img_mask = cv2.imread(pair_images[1]['mask'],-1)
        if len(query_img_mask.shape) == 3:
            query_img_mask = self.color2digit(query_img_mask,self.mapMatrix)
            query_img_mask = ((query_img_mask==2)*1).astype(np.float32)
        else:
            query_img_mask = self.transform(query_img_mask)
        # print(np.unique(query_img_mask),pair_images[1]['mask'])

        return support_img, support_img_mask, query_img, query_img_mask


class SarShipTest(Dataset):

    def __init__(self,root_path, data_class, transform=transform,mode='test',k_shot=1):

        self.root_path = root_path
        self.mode = mode
        self.data_classes = data_class

        print('{} on {}'.format(self.mode,  self.data_classes))
        self.group_images,self.total_list = self.get_group_image()
        self.transform = transform
        self.mapMatrix = self.mapcolor()
        self.k_shot = k_shot
        self.count = 0
        # To ensure to result can be reproduced
        self.random_generator = random.Random()
        self.random_generator.seed(24)

    def mapcolor(self):
        '''
        read images by opencv:the channel order : [B,G,R]
        '''
        colormap = [[240, 241, 242],
                    [255, 218, 170],
                    [174, 136, 192]]

        mapMatrix = np.zeros(256 * 256 * 256, dtype=np.int32)
        for i, cm in enumerate(colormap):
            mapMatrix[cm[0] * 65536 + cm[1] * 256 + cm[2]] = i

        return mapMatrix

    def color2digit(self, label, mapMatrix):

        data = label.astype('int32')
        index = data[:, :, 0] * 65536 + data[:, :, 1] * 256 + data[:, :, 2]
        digital = mapMatrix[index]

        return digital

    def get_group_image(self):

        total_img_list = []
        group_images = dict()
        for data_class in self.data_classes:
            group_images[data_class] = []
            for data in os.listdir(os.path.join(self.root_path,data_class,'data')):
                item = {}
                item['data'] = os.path.join(os.path.join(self.root_path,data_class,'data',data))
                item['mask'] = os.path.join(os.path.join(self.root_path,data_class,'mask',data))
                group_images[data_class].append(item)
                total_img_list.append([data,data_class])
    
        print("Total images :",len(total_img_list))
                
        return group_images, total_img_list
    
    def get_one_shot(self,idx):

        if self.count >= len(self.total_list):
            self.random_generator.shuffle(self.total_list)
            self.count = 0

        query_name,class_ = self.total_list[self.count]
        
        while True:
            support_img_list = self.group_images[class_]
            support_meta = support_img_list[self.random_generator.randint(0,len(support_img_list)-1)]
            if support_meta['data'].split('/')[-1] != query_name:
                break
        # for support image
        support_img = cv2.imread(support_meta['data'],cv2.IMREAD_COLOR)
        support_img = self.transform(support_img)
        support_img_mask = cv2.imread(support_meta['mask'],-1)
        if len(support_img_mask.shape) == 3:
            support_img_mask = self.color2digit(support_img_mask,self.mapMatrix)
            support_img_mask = ((support_img_mask==2)*1).astype(np.float32)
        else:
            support_img_mask = self.transform(support_img_mask)

        # print('==suport',np.unique(support_img_mask),pair_images[0]['mask'])
        # for query image
        query_img = cv2.imread(os.path.join(self.root_path,class_,'data',query_name),cv2.IMREAD_COLOR)
        query_img = self.transform(query_img)
        
        query_img_mask = cv2.imread(os.path.join(self.root_path,class_,'mask',query_name),-1)
        if len(query_img_mask.shape) == 3:
            query_img_mask = self.color2digit(query_img_mask,self.mapMatrix)
            query_img_mask = ((query_img_mask==2)*1).astype(np.float32)
        else:
            query_img_mask = self.transform(query_img_mask)

        self.count += 1

        return support_img,support_img_mask,query_img,query_img_mask

    def get_k_shot(self,idx):

        if self.count >= len(self.total_list):
            self.random_generator.shuffle(self.total_list)
            self.count = 0
        query_name,class_ = self.total_list[self.count]
        
        support_set_list = self.group_images[class_]
        support_choice_list = support_set_list.copy()
        support_choice_list = [meta for meta in support_choice_list if meta['data'].split('/')[-1] != query_name]
        support_meta_list = self.random_generator.sample(support_choice_list,self.k_shot)

        # for support images
        support_img_list,support_img_mask_list = [],[]
        for meta in support_meta_list:
            support_img = cv2.imread(meta['data'],cv2.IMREAD_COLOR)
            support_img = self.transform(support_img)
            support_img_mask = cv2.imread(meta['mask'],-1)
            if len(support_img_mask.shape) == 3:
                support_img_mask = self.color2digit(support_img_mask,self.mapMatrix)
                support_img_mask = ((support_img_mask==2)*1).astype(np.float32)
            else:
                support_img_mask = self.transform(support_img_mask)

            support_img_list.append(support_img)
            support_img_mask_list.append(support_img_mask)
        
        for i in range(len(support_img_list)):
            support_temp_img = support_img_list[i]
            support_temp_mask = support_img_mask_list[i][None,:]
            if i == 0:
                support_img = support_temp_img
                support_mask = support_temp_mask
            else:
                support_img = np.concatenate([support_img,support_temp_img],axis=0)
                support_mask = np.concatenate([support_mask,support_temp_mask],axis=0)

        # for query image
        query_img = cv2.imread(os.path.join(self.root_path,class_,'data',query_name),cv2.IMREAD_COLOR)
        query_img = self.transform(query_img)
        
        query_img_mask = cv2.imread(os.path.join(self.root_path,class_,'mask',query_name),-1)
        if len(query_img_mask.shape) == 3:
            query_img_mask = self.color2digit(query_img_mask,self.mapMatrix)
            query_img_mask = ((query_img_mask==2)*1).astype(np.float32)
        else:
            query_img_mask = self.transform(query_img_mask)

        self.count += 1

        return support_img,support_mask,query_img,query_img_mask


    def __len__(self):

        return len(self.total_list)

    
    def __getitem__(self,idx):

        if self.k_shot == 1:
            support_img, support_img_mask, query_img, query_img_mask = self.get_one_shot(idx)
        else:
            support_img, support_img_mask, query_img, query_img_mask = self.get_k_shot(idx)
        
        return support_img, support_img_mask, query_img, query_img_mask
            