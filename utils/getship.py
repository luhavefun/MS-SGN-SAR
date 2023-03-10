import os
import shutil
import numpy as np
import cv2
from tqdm import tqdm

def checkship(mask):
    '''
    In my dataset,there are only three labels:[0,1,2]
    0->land, 1->sea, 2->ship
    '''
    labels = list(np.unique(mask))
    if 2 in labels:
        return True
    else:
        return False


def get_ship(src,dst):

    folds = os.listdir(src)
    for fold in folds:
        mask_paths = [os.path.join(src,fold,'first_exp/data_augment/mask',f) \
                    for f in os.listdir(os.path.join(src,fold,'first_exp/data_augment/mask'))]
        for path in tqdm(mask_paths):
            mask = cv2.imread(path)
            if checkship(mask):
                shutil.copy(path,os.path.join(dst,fold,'mask'))
                datapath = os.path.join(src,fold,'first_exp/data_augment/data',os.path.split(path)[-1])
                shutil.copy(datapath,os.path.join(dst,fold,'data'))

if __name__ == '__main__':

    src = r'E:\ship_segmentation_multiclass\firstjob'
    dst = r'E:\ship_segmentation_multiclass\fewshot'
    get_ship(src,dst)







