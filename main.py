import torch
from torch import nn,optim
from torch.nn import functional as F
from torch.utils.data import DataLoader,random_split
import os
import numpy as np
import argparse
from utils import FCN8s,FCN16s,FCN32s,FCN8x,train_transform,voc_dataset,label_accuracy_score,label2image
import matplotlib.pyplot as plt
import random

torch.manual_seed(78)
data_path='./data' #数据集路径
result_path='./result.txt' #结果打印路径
model_path='./models/best_model.mdl' #最优模型保存路径
bs=5
lr=0.01
epoch=10
numclasses=21
crop=(256,256)
ratio=0.9 #训练集比例
use_gpu=torch.cuda.is_available()

if os.path.exists(result_path):
    os.remove(result_path)

#构建数据集
dataset=voc_dataset(root=data_path,transfrom=train_transform,crop_size=crop)
train_size=int(len(dataset)*ratio)
train_data,val_data=random_split(dataset,[train_size,len(dataset)-train_size])
train_dataloader=DataLoader(train_data,batch_size=bs,shuffle=True)
val_dataloader=DataLoader(val_data,batch_size=bs,shuffle=False)

#构建网络
net=FCN8s(numclasses)
optimizer=optim.SGD(net.parameters(),lr=lr,weight_decay=1e-4)
criterion=nn.NLLLoss()
if use_gpu:
    net.cuda()
    criterion=criterion.cuda()

#训练验证
def train():
    best_score=0.0
    for e in range(epoch):
        net.train()
        train_loss=0.0
        label_true=torch.LongTensor()
        label_pred=torch.LongTensor()
        for i,(batchdata,batchlabel) in enumerate(train_dataloader):
            '''
            batchdata:[b,3,h,w] c=3
            batchlabel:[b,h,w] c=1 直接去掉了
            '''
            if use_gpu:
                batchdata,batchlabel=batchdata.cuda(),batchlabel.cuda()

            output=net(batchdata)
            output=F.log_softmax(output,dim=1)
            loss=criterion(output,batchlabel)

            pred=output.argmax(dim=1).squeeze().data.cpu()
            real=batchlabel.data.cpu()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss+=loss.cpu().item()*batchlabel.size(0)
            label_true=torch.cat((label_true,real),dim=0)
            label_pred=torch.cat((label_pred,pred),dim=0)

        train_loss/=len(train_data)
        acc, acc_cls, mean_iu, fwavacc=label_accuracy_score(label_true.numpy(),label_pred.numpy(),numclasses)

        print('\n epoch:{}, train_loss:{:.4f}, acc:{:.4f}, acc_cls:{:.4f}, mean_iu:{:.4f}, fwavacc:{:.4f}'.format(
            e+1,train_loss,acc, acc_cls, mean_iu, fwavacc))

        with open(result_path, 'a') as f:
            f.write('\n epoch:{}, train_loss:{:.4f}, acc:{:.4f}, acc_cls:{:.4f}, mean_iu:{:.4f}, fwavacc:{:.4f}'.format(
                e+1,train_loss,acc, acc_cls, mean_iu, fwavacc))

        net.eval()
        val_loss=0.0
        val_label_true = torch.LongTensor()
        val_label_pred = torch.LongTensor()
        with torch.no_grad():
            for i,(batchdata,batchlabel) in enumerate(val_dataloader):
                if use_gpu:
                    batchdata,batchlabel=batchdata.cuda(),batchlabel.cuda()

                output=net(batchdata)
                output=F.log_softmax(output,dim=1)
                loss=criterion(output,batchlabel)

                pred = output.argmax(dim=1).squeeze().data.cpu()
                real = batchlabel.data.cpu()

                val_loss+=loss.cpu().item()*batchlabel.size(0)
                val_label_true = torch.cat((val_label_true, real), dim=0)
                val_label_pred = torch.cat((val_label_pred, pred), dim=0)

            val_loss/=len(val_data)
            val_acc, val_acc_cls, val_mean_iu, val_fwavacc=label_accuracy_score(val_label_true.numpy(),
                                                                                val_label_pred.numpy(),numclasses)
        print('\n epoch:{}, val_loss:{:.4f}, acc:{:.4f}, acc_cls:{:.4f}, mean_iu:{:.4f}, fwavacc:{:.4f}'.format(
            e+1,val_loss,val_acc, val_acc_cls, val_mean_iu, val_fwavacc))

        with open(result_path, 'a') as f:
            f.write('\n epoch:{}, val_loss:{:.4f}, acc:{:.4f}, acc_cls:{:.4f}, mean_iu:{:.4f}, fwavacc:{:.4f}'.format(
            e+1,val_loss,val_acc, val_acc_cls, val_mean_iu, val_fwavacc))

        score=(val_acc_cls+val_mean_iu)/2
        if score>best_score:
            best_score=score
            torch.save(net.state_dict(),model_path)

#加载网络进行测试
def evaluate():
    net.load_state_dict(torch.load(model_path))
    index = random.randint(0, len(dataset) - 1)
    val_image, val_label = dataset[index]

    out = net(val_image.unsqueeze(0).cuda())
    pred = out.argmax(dim=1).squeeze().data.cpu().numpy()
    label = val_label.data.numpy()
    val_pred, val_label = label2image(numclasses)(pred, label)

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(val_image.numpy().transpose(1, 2, 0).astype('uint8'))
    ax[1].imshow(val_label)
    ax[2].imshow(val_pred)
    plt.show()

## 使用argparse示例
# def main(args):
#     Net = None
#     if args.model == 'fcn8':
#         Net = FCN8
#     if args.model == 'fcn16':
#         Net = FCN16
#     if args.model == 'fcn32':
#         Net = FCN32
#     assert Net is not None, f'model {args.model} not available'
#
#     model = Net(NUM_CLASSES)
#
#     if args.cuda:
#         model = model.cuda()
#     if args.state:
#         try:
#             model.load_state_dict(torch.load(args.state))
#         except AssertionError:
#             model.load_state_dict(torch.load(args.state,
#                 map_location=lambda storage, loc: storage))
#
#     if args.mode == 'eval':
#         evaluate(args, model)
#     if args.mode == 'train':
#         train(args, model)

if __name__ == '__main__':
    # parse=argparse.ArgumentParser(description='FCN')
    # parse.add_argument('--model',required=True)
    # parse.add_argument('--state')
    #
    # subparse=parse.add_subparsers(dest='mode')
    #
    # parser_eval=subparse.add_parser('eval')
    # parser_eval.add_argument('--image')
    # parser_eval.add_argument('--label')
    #
    # parser_train=subparse.add_parser('train')
    # parser_train.add_argument('--bs',type=int,default=5)
    # parser_train.add_argument('--epoch',type=int,default=10)
    # parser_train.add_argument('--lr',type=float,default=0.001)
    #
    # args=parse.parse_args()

    train()
