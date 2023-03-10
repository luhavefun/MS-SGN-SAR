import os
import torch
import json
import argparse
import cv2
# from oneshot_model import OneShotModel
# from oneshot_resunet import ResUnetOneShot
from networks.ResUnetFRN import ResUnetOneShot
from dataset import SarShipTest
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

ROOT_PATH = os.getcwd()
print(ROOT_PATH)
SNAPSHOT_DIR = os.path.join(ROOT_PATH, 'snapshots')
IMG_DIR = os.path.join(ROOT_PATH, 'datasets')
# IMG_DIR = os.path.join(ROOT_PATH, 'inshore-offshore-data')

RESULT_DIR = os.path.join(ROOT_PATH, 'result')


def get_arguments():
    parser = argparse.ArgumentParser(description='OneShot Segmentation Test of Sar Ship')
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR)
    parser.add_argument("--result", type=str, default=RESULT_DIR)
    parser.add_argument("--img_dir", type=str, default=IMG_DIR)
    parser.add_argument("--group",type=int,default=0)
    parser.add_argument('--restore_step', type=int, default=30000)
    parser.add_argument("--batch_size",type=int,default=1)
    parser.add_argument("--test_class",nargs="+",required=True)
    parser.add_argument("--k_shot",type=int,default=1)
    parser.add_argument("--vis",dest='vis',action='store_true')

    return parser.parse_args()


def measure(y_in, pred_in):
    # thresh = .5
    thresh = .5
    y = y_in > thresh
    pred = pred_in > thresh
    tp = np.logical_and(y, pred).sum()
    tn = np.logical_and(np.logical_not(y), np.logical_not(pred)).sum()
    fp = np.logical_and(np.logical_not(y), pred).sum()
    fn = np.logical_and(y, np.logical_not(pred)).sum()
    return tp, tn, fp, fn


def restore(args, model):
    savedir = os.path.join(args.snapshot_dir, 'group_%d'% args.group)
    # filename = 'step_%d.pth.tar' % (args.restore_step)
    filename = 'step_%d.pth.tar' % (args.restore_step)
    snapshot = os.path.join(savedir, filename)
    assert os.path.exists(snapshot), "Snapshot file %s does not exist." % (snapshot)

    checkpoint = torch.load(snapshot)
    model.load_state_dict(checkpoint['state_dict'])

    print('Loaded weights from %s' % (snapshot))


def viewsar(img):
    '''
    The orgin sar image is uint16 format, to view it formly by changing to uint8
    '''
    img = np.squeeze(img,axis=0)
    img = img[0,:,:]*255.0

    img_g = img > 255
    img_l = img <= 255
    img_v = img_g * 255 + img_l * img

    return img_v


def pairshow(sup_img, query_img, query_true, query_pred, filepath):

    fig = plt.gcf()
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.9, left=0.1, hspace=0, wspace=0.1)
    plt.margins(0, 0)

    plt.subplot(141)
    plt.axis('off')
    plt.title('support image', fontsize=10, style='italic', fontweight='bold')
    # plt.imshow(viewsar(sup_img), cmap='gray')
    sup_img = np.squeeze(sup_img,axis=0)
    sup_img = np.transpose(sup_img,(1,2,0))
    plt.imshow(sup_img[:,:,0],cmap=plt.cm.gray,interpolation='nearest')

    plt.subplot(142)
    plt.axis('off')
    plt.title('query image', fontsize=10, style='italic', fontweight='bold')
    # plt.imshow(viewsar(query_img), cmap='gray')
    
    query_img = np.squeeze(query_img,axis=0)
    query_img = np.transpose(query_img,(1,2,0))
    plt.imshow(query_img[:,:,0],cmap=plt.cm.gray,interpolation='nearest')

    plt.subplot(143)
    plt.axis('off')
    plt.title('query gt', fontsize=10, style='italic', fontweight='bold')
    plt.imshow(query_true, cmap='gray')

    plt.subplot(144)
    plt.axis('off')
    plt.title('query pred', fontsize=10, style='italic', fontweight='bold')
    plt.imshow(query_pred, cmap='gray')

    plt.show()
    fig.savefig(filepath, format='png', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0)


def save_pair_image(support_img,query_img,query_mask,query_pred,filename):

    support_img = viewsar(support_img)
    query_img = viewsar(query_img)

    res = np.zeros(( support_img.shape[1],support_img.shape[0]*4),dtype=np.uint8)
    res[:, 0:512] = support_img
    res[:,512:512*2] = query_img
    res[:,512*2:512*3] = query_mask*255
    res[:,512*3:] = query_pred*255

    cv2.imwrite(filename,res)

    # res = Image.new(mode='L',size=(support_img.shape[0]*4, support_img.shape[1]))
    # print(res.size)
    # res.paste(support_img, (0,0,512,512))
    # res.paste(query_img,box=(512,0))
    # res.paste(query_mask,box=(512*2,0))
    # res.paste(query_pred,box=(512*3,0))
    # res.save(filename)


def test(args):
    model = ResUnetOneShot()
    model = model.cuda()
    model.eval()

    restore(args, model)

    eps = 0.000001
    ious = []
    precisions = []
    recalls = []
    fscores = []
    sar_test_set = SarShipTest(args.img_dir,data_class=args.test_class,mode='test')
    test_loader = DataLoader(sar_test_set, batch_size=args.batch_size, shuffle=False)
    count = 0

    for data in test_loader:
        begin_time = time.time()
        
        support_img, support_img_mask, query_img, query_img_mask = data
        support_img, support_img_mask, query_img, query_img_mask = \
            support_img.cuda(), support_img_mask.cuda(), query_img.cuda(), query_img_mask.cuda()

        # support_img = support_img * support_img_mask
        support_imgo = support_img
        support_img = torch.mul(support_img,support_img_mask.unsqueeze(1))
        # logits = model(support_img, support_img_mask, query_img)

        with torch.no_grad():
            logits = model(support_img, query_img)
            pred = model.get_pred(logits)
            pred = pred.data.cpu().numpy().astype(np.int8)
            query_img_mask = query_img_mask.cpu().numpy().astype(np.uint8)


        for i in range(args.batch_size):
            tp, tn, fp, fn = measure(pred[i,:,:], query_img_mask[i,:,:])
            iou = tp / (tp + fp + fn)
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f_score = 2 * precision * recall/(precision + recall + eps)
            
            ious.append(iou)
            precisions.append(precision)
            recalls.append(recall)
            fscores.append(f_score)
            end_time = time.time()
            count += 1
            imgPersec = 1/(end_time - begin_time)
            print('It  has detected %d,%.2f images/s'%(count,imgPersec),end = "\r")
            
            # print("support_img shape",support_img.shape)
            # print("query_img",query_img.shape)
            # print("pred shape",pred.shape)
            if args.vis:
                res_root = os.path.join(os.getcwd(),'result')
                assert os.path.exists(res_root),'PATH NOT EXISTS!!!'
                pairshow(sup_img=support_imgo.cpu().numpy().astype(np.uint8),query_img=query_img.cpu().numpy().astype(np.uint8),
                         query_true=query_img_mask[i,:,:],query_pred=pred[i,:,:],filepath=os.path.join(res_root,'{}.png'.format(count)))
                if count > 10:
                    break
        # add some code to show result
        # res_root = os.path.join(args.result,'showimage')
        # if not os.path.exists(res_root):
        #     os.makedirs(res_root)
        # pairshow(sup_img=support_img.cpu().numpy(),query_img=query_img.cpu().numpy(),
        #         query_true=query_img_mask,query_pred=pred,filepath=os.path.join(res_root,'{}.png'.format(count)))
        # if count > 5:
        #     break

        # add some code to save pair image
        # res_root = os.path.join(args.result,'res_HK')
        # if not os.path.exists(res_root):
        #     os.makedirs(res_root)
        # save_pair_image(support_img=support_img.cpu().numpy(),query_img=query_img.cpu().numpy(),
        #         query_mask=query_img_mask,query_pred=pred,filename=os.path.join(res_root,'{}.png'.format(count)))

    print('All episodes have been processed')
    print('The Mean Iou is: %.4f' % np.mean(ious))
    print('The Mean precison is: %.4f' % np.mean(precisions))
    print('The Mean recall is: %.4f' % np.mean(recalls))
    print('The Mean f1-socre is: %.4f' % np.mean(fscores))

    # return np.mean(ious),np.mean(precisions),np.mean(recalls),np.mean(fscores)


if __name__ == '__main__':

    args = get_arguments()
    print('Runing Test parameters:\n')
    print(json.dumps(vars(args), indent=4, separators=(',', ':')))
    test(args)

