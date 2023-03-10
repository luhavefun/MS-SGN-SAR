import os
import torch
import json
import argparse
import shutil

from networks.ResUnet import ResUnetOneShot
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from dataset import SarShipTrain
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import torch


ROOT_DIR = os.getcwd()
IMG_DIR = os.path.join(ROOT_DIR,'datasets')
SNAPSHOT_DIR = os.path.join(ROOT_DIR,'snapshots')
PRETRAINED = os.path.join(ROOT_DIR,'snapshots/pretrained/vgg16-397923af.pth')
# print(ROOT_DIR)
# print(IMG_DIR)
# print(SNAPSHOT_DIR)

def get_arguments():
    
    parser = argparse.ArgumentParser(description='OneShot Segmentation of Sar Ship')
    
    parser.add_argument("--max_steps", type=int, default=60001)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--disp_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=10000)
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR)
    parser.add_argument("--img_dir",type=str,default=IMG_DIR)
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--start_count", type=int, default=0)
    parser.add_argument("--pretrained_weight",type=str,default=PRETRAINED)
    parser.add_argument('--val_interval',type=int,default=2000)

    return parser.parse_args()

class AverageMeter(object):

    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self,val,n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(args,state,is_best,filename='checkpoint.pth.tar'):
    save_dir = os.path.join(args.snapshot_dir,'resunet/test_SH')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_path = os.path.join(save_dir,filename)
    torch.save(state,save_path)
    if is_best:
        shutil.copy(save_path,os.path.join(args.snapshot_dir,'model_best.pth.tar'))

def restore(snapshot_dir,model):

    filelist = os.listdir(snapshot_dir)
    filelist = [x for x in filelist if os.path.isfile(os.path.join(snapshot_dir,x)) and x.endswith('.pth.tar')]
    if len(filelist)>0:
        filelist.sort(key=lambda x:os.path.getmtime(os.path.join(snapshot_dir,x)),reverse=True)
        snapshot = os.path.join(snapshot_dir,filelist[0])
    else:
        snapshot = ''
    
    if os.path.isfile(snapshot):
        print("=====> loading checkpoint '{}'".format(snapshot))
        checkpoint = torch.load(snapshot)
        try:
            model.load_state_dict(checkpoint['state_dict'])
            print("=====> loading model from '{}'".format(snapshot))
        except KeyError:
            raise Exception('Loading pre-trained model failed')
    
    else:
        raise Exception("No checkpoints found at'{}'".format(snapshot))

def get_model_parameter_num(model):
    '''
    compute the total parameters of model
    '''
    total_num = 0
    for param in model.parameters():
        total_num += torch.numel(param)
    
    return total_num

def measure(y_pred,y_true):

    thresh = 0.5
    e = 0.00001
    y_pred = y_pred > thresh
    y_true = y_true > thresh

    tp = np.logical_and(y_true,y_pred).sum()
    tn = np.logical_and(np.logical_not(y_true),np.logical_not(y_pred)).sum()
    fp = np.logical_and(np.logical_not(y_true),y_pred).sum()
    fn = np.logical_and(y_true,np.logical_not(y_pred)).sum()

    precision = tp/(tp + fp + e)
    recall = tp/(tp + fn)

    return precision,recall

def freeze_bn(m):
    if isinstance(m, nn.BatchNorm2d):
        m.eval()

def train(args):

    # model = OneShotModel()
    model = ResUnetOneShot()
    model = model.cuda()
    model.train()
    print('Number of Parameters %d'%get_model_parameter_num(model))
    optimizer = optim.SGD(params=model.parameters(),lr=args.lr,momentum=0.9,weight_decay=1e-4)
    
    if args.resume:
        restore(args.snapshot_dir,model)
        print("Resume training .............")
    
    lr_ploy = lambda step: (1-float(step)/args.max_steps)**0.9
    scheduler = LambdaLR(optimizer,lr_lambda=lr_ploy)

    train_data = SarShipTrain(root_path=args.img_dir,data_class=['QD','HK'],mode='train')
    train_loader = DataLoader(train_data,batch_size=1,shuffle=False,num_workers=4)

    # val_data = SarShipTrain(root_path=args.img_dir,data_class=['SH'],mode='val')
    # val_loader = DataLoader(val_data,batch_size=1,shuffle=False,num_workers=4)

    count = args.start_count

    loss = AverageMeter()
    precision = AverageMeter()
    recall = AverageMeter()
    F1_Score = AverageMeter()
    best_f1 = 0.

    for data in train_loader:
        if count > args.max_steps:
            print(" Training step have got the max value ")
            print('Best F1-score:', best_f1)

            break
        support_img, support_img_mask, query_img, query_img_mask = data
        support_img, support_img_mask, query_img, query_img_mask = \
                    support_img.cuda(), support_img_mask.cuda(), query_img.cuda(), query_img_mask.cuda()

        # logits = model(support_img, support_img_mask, query_img)
        support_img = support_img * support_img_mask #
        logits = model(support_img,query_img)

        pred = model.get_pred(logits)
        # print(pred)
        # print(query_img_mask)
        pred = pred.data.cpu().numpy().astype(np.int32)

        celoss = model.get_celoss(logits,query_img_mask)
        query_img_mask = query_img_mask.cpu().numpy().astype(np.int32).squeeze(0)

        precision_, recall_ = measure(pred, query_img_mask)
        f1_score_ = 2*(precision_*recall_) /(precision_ + recall_+1e-10)

        loss.update(celoss.data.item())
        # precision.update(precision_.data.item())
        # recall.update(recall_.data.item())
        precision.update(precision_)
        recall.update(recall_)
        F1_Score.update(f1_score_)

        optimizer.zero_grad()
        celoss.backward()
        optimizer.step()

        count += 1
        scheduler.step()

        is_best = f1_score_ > best_f1
        best_f1 = max(f1_score_,best_f1)

        if count % args.disp_interval == 0:
            print('Step:%d \t Loss:%.4f \t Precision:%.4f \t Recall:%.4f \t F1-Score:%.4f'%(count,loss.avg, precision.avg, recall.avg,F1_Score.avg))

        # if count % args.val_interval == 0:
        #     f1_scores = []
        #     for valset in val_loader:
        #         val_support_img, val_support_img_mask, val_query_img, val_query_img_mask = valset
        #         val_support_img, val_support_img_mask, val_query_img, val_query_img_mask = \
        #             val_support_img.cuda(), val_support_img_mask.cuda(), val_query_img.cuda(), val_query_img_mask.cuda()

        #         val_support_img = val_support_img * val_support_img_mask
        #         logits = model(val_support_img, val_query_img)

        #         pred = model.get_pred(logits)
        #         pred = pred.data.cpu().numpy().astype(np.uint8)
        #         query_img_mask = val_query_img_mask.cpu().numpy().astype(np.uint8).squeeze()
        #         val_precison, val_recall = measure(pred, query_img_mask)
        #         F1 = 2*(val_precison*val_recall)/(val_precison + val_recall + 1e-10)
        #         f1_scores.append(F1)
        #     print("===="*20)
        #     print('After {} steps,the f1-score on valdata is {}'.format(count,np.mean(f1_scores)))

        if count % args.save_interval==0:
            save_checkpoint(args,
                            {'global_counter': count,
                             'state_dict':model.state_dict(),
                             'optimizer':optimizer.state_dict()}, 
                            is_best=is_best,
                            filename='step_%d.pth.tar'%(count))

if __name__ == "__main__":

    args = get_arguments()
    print('Runing parameters:\n')
    print(json.dumps(vars(args),indent=4,separators=(',',':')))

    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)
    train(args)