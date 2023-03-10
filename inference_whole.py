import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import argparse
from oneshot_resunet_frn import ResUnetOneShot
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def get_args():

    parser = argparse.ArgumentParser(description='Few shot test on Whole SAR Image')
    parser.add_argument('--img_path',type=str,default='datasets/four_area_uint8/HK.tif')
    parser.add_argument('--patch_h',type=int,default=512)
    parser.add_argument('--patch_w',type=int,default=512)
    parser.add_argument('--stride',type=int,default=256)
    parser.add_argument('--sup_img_root',type=str,default='datasets/HK/data')
    parser.add_argument('--sup_mask_root',type=str,default='datasets/HK/mask')
    parser.add_argument('--weight_path',type=str,default='snapshots/resunet/test_HK_FRN/step_30000_best.pth.tar')
    parser.add_argument('--res_root',type=str,default='result/HK')
    parser.add_argument('--area',type=str,help='the segmentation save name',default='HK')
    args = parser.parse_args()
    
    return args

def mapcolor():
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


def color2digit(label, mapMatrix):

    data = label.astype('int32')
    index = data[:, :, 0] * 65536 + data[:, :, 1] * 256 + data[:, :, 2]
    digital = mapMatrix[index]

    return digital

def label2color(segm):

    h = segm.shape[0]
    w = segm.shape[1]

    label = np.zeros((h, w, 3), dtype=np.uint8)
    color_label = np.zeros((h, w, 3), dtype=np.uint8)
    _, color_map = mapcolor()
    cl = np.unique(segm)
    for i, cm in enumerate(cl):
        flag = (segm == cm)
        label[:, :, 0] = flag * color_map[cm][0]
        label[:, :, 1] = flag * color_map[cm][1]
        label[:, :, 2] = flag * color_map[cm][2]

        color_label[:, :, 0] += label[:, :, 0]
        color_label[:, :, 1] += label[:, :, 1]
        color_label[:, :, 2] += label[:, :, 2]
    return color_label


def get_random_sup(img_root_path,mask_root_path,k_shot=1):

    filenames = os.listdir(img_root_path)
    sup_filenames = np.random.choice(filenames,k_shot,replace=False)
    sup_meta = []
    for sup_fileanme in sup_filenames:
        sup_img = cv2.imread(os.path.join(img_root_path,sup_fileanme),cv2.IMREAD_COLOR)/255.0
        h,w,c = sup_img.shape
        sup_img = sup_img.transpose(2,0,1).astype(np.float32)

        sup_mask = cv2.imread(os.path.join(mask_root_path,sup_fileanme),cv2.IMREAD_UNCHANGED)
        sup_mask_digital = color2digit(sup_mask, mapcolor())
        print(np.unique(sup_mask_digital))
        sup_mask = ((sup_mask_digital==2)*1).astype(np.float32) # change to 1 for sea

        sup_meta.append(sup_img * sup_mask)

    sup_meta_array = np.array(sup_meta).reshape((k_shot,1,c,h,w))
    sup_meta_array = torch.from_numpy(sup_meta_array)

    return sup_meta_array

class SarQuerySet(Dataset):

    def __init__(self,img_path,patch_h,path_w,stride):
        
        self.full_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        self.patch_h = patch_h
        self.patch_w = path_w
        self.stride = stride
        self.rows = self.full_img.shape[0] // self.stride
        self.columns = self.full_img.shape[1] // self.stride
        self.pad_img = self.get_pad(self.full_img)
        
    def get_pad(self,img):

        padding_h = self.rows * self.stride + self.patch_h
        padding_w = self.columns * self.stride + self.patch_w
        padding_img = np.zeros((padding_h,padding_w,3),dtype=np.float32)

        padding_img[0:img.shape[0], 0:img.shape[1],:] = img

        return padding_img
    
    def __len__(self):

        row_iters = self.full_img.shape[0] // self.stride
        column_iters = self.full_img.shape[1] // self.stride

        return row_iters * column_iters

    def __getitem__(self,index):

        assert index < len(self), 'Index Error!!!'

        col_index = index // self.rows
        row_index = index % self.rows

        ly = row_index * self.stride
        lx = col_index * self.stride

        query_patch = self.pad_img[ly:ly+self.patch_h, lx:lx+self.patch_w,:]
        query_patch = query_patch.transpose(2,0,1) /255.0

        return torch.from_numpy(query_patch)

def kshot_inference(args):

    query_set = SarQuerySet(args.img_path, args.patch_h, args.patch_w, args.stride)
    query_loader = DataLoader(query_set, batch_size=1, shuffle=False)

    sup_img = get_random_sup(args.sup_img_root, args.sup_mask_root,k_shot=5)
    sup_img = sup_img.cuda()

    model = ResUnetOneShot()
    model = model.cuda()
    print('======== load weight from {} ======='.format(args.weight_path))
    checkpoint = torch.load(args.weight_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print('=========== model fixed =============')
    res = np.zeros((query_set.pad_img.shape[0], query_set.pad_img.shape[1], 2), dtype=np.float32)

    print('The total patches:', len(query_loader))

    for i, query_data in enumerate(query_loader):
        query_data = query_data.cuda()
        temp = []
        for k in range(sup_img.shape[0]):
            segms = model(sup_img[k], query_data)
            segms = segms.squeeze(0)
            segms = segms.permute(1, 2, 0)

            temp.append(segms.cpu().detach().numpy())

        predict = sum(temp)

        col_index = i // query_set.rows
        row_index = i % query_set.rows

        res[row_index * query_set.stride:row_index * query_set.stride + query_set.patch_h, \
        col_index * query_set.stride:col_index * query_set.stride + query_set.patch_w,
        :] += predict

        print('The {}th has been processed'.format(i))

    digital_label = np.argmax(res, axis=-1)
    digital_label = np.array(digital_label * 255, dtype=np.uint8)

    file_name = '{}_5shot_land.tif'.format(args.area)
    seg_res = digital_label[0:query_set.full_img.shape[0], 0:query_set.full_img.shape[1]]

    if not os.path.exists(args.res_root):
        os.makedirs(args.res_root)

    cv2.imwrite(os.path.join(args.res_root, file_name), seg_res)

    print('=== the segmentation result saved in {} ==='.format(args.res_root))
    print('========= Full Image has been processed and saved ============')

def main(args):

    query_set = SarQuerySet(args.img_path, args.patch_h, args.patch_w, args.stride)
    query_loader = DataLoader(query_set, batch_size=1, shuffle=False)

    sup_img = get_random_sup(args.sup_img_root, args.sup_mask_root)
    sup_img = sup_img.cuda()

    model = ResUnetOneShot()
    model = model.cuda()
    print('======== load weight from {} ======='.format(args.weight_path))
    checkpoint = torch.load(args.weight_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print('=========== model fixed =============')
    res = np.zeros((query_set.pad_img.shape[0], query_set.pad_img.shape[1], 2),dtype=np.float32)

    print('The total patches:',len(query_loader))
    for i, query_data in enumerate(query_loader):
        query_data = query_data.cuda()
        segms = model(sup_img,query_data)
        segms = segms.squeeze(0)
        segms = segms.permute(1,2,0)

        col_index = i // query_set.rows
        row_index = i % query_set.rows

        res[row_index*query_set.stride:row_index*query_set.stride+query_set.patch_h,\
           col_index*query_set.stride:col_index*query_set.stride+query_set.patch_w,:] += segms.cpu().detach().numpy()

        print('The {}th has been processed'.format(i))
    
    digital_label = np.argmax(res,axis=-1)
    digital_label = np.array(digital_label*255,dtype=np.uint8)

    file_name = '{}_oneshot_land.tif'.format(args.area)
    seg_res = digital_label[0:query_set.full_img.shape[0],0:query_set.full_img.shape[1]]

    if not os.path.exists(args.res_root):
        os.makedirs(args.res_root)

    cv2.imwrite(os.path.join(args.res_root,file_name),seg_res)

    print('=== the segmentation result saved in {} ==='.format(args.res_root))
    print('========= Full Image has been processed and saved ============')

        
if __name__ == '__main__':

    args = get_args()
    kshot_inference(args)

























        




