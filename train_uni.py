#from eval_IOU import *
import torch.backends.cudnn as cudnn
import numpy as np 
import torch 
import torch.nn as nn
from utils.criterion import CrossEntropyLoss2d
from data_loader import LIP
from model.resnet_tiny_refine import resnet_tiny_refine
from model.BiSEnet import BiSeNet
from model.EDAnet import EDANet
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import ConcatDataset
from model.deeplabv2 import *
from utils.eval_IOU import evaluate
from tensorboardX import SummaryWriter
from utils.matting_loss import Matting_loss

writer = SummaryWriter('./log/res50/')

from config import *
cudnn.enabled = True
cudnn.benchmark = True

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
np.random.seed(1)



def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))


def lr_warm_up(base_lr,iter,max_iter,power,up,keep):
    if iter < up:
        return base_lr * ( (  float(iter)/up ) ** power )
    if iter >= up and iter < keep:
        return base_lr
    if iter >= keep:
        return base_lr * ((1-float(iter-keep)/(max_iter-keep) )**(power))



def adjust_learning_rate(optimizer, i_iter):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    #lr = lr_poly(Lr, i_iter, NUM_STEPS, POWER)

    lr = lr_warm_up(Lr, i_iter, NUM_STEPS, POWER, UP, KEEP)
    #lr = lr_poly(Lr,iter,NUM_STEPS,1)
    for param_lr in optimizer.param_groups:
        param_lr['lr'] = lr
    return lr



#------model--------------
model = resnet_tiny_refine().cuda()
#model = EDANet().cuda()
#model = ResNet(Bottleneck , [3, 4, 6, 3], 20).cuda()


#-------------data--------------------


train_data= LIP()
train_loader = DataLoader(train_data, BATCH_SIZE,shuffle=True,num_workers=8,pin_memory=True,drop_last=True)
print('len of train loader: ', len(train_loader))



optimizer = torch.optim.SGD(filter(lambda p :p.requires_grad,model.parameters()),lr=Lr,momentum=0.9,weight_decay=0.0001)
weight = torch.Tensor( [0.5,2,1,1,3,3]).cuda()
criterion = nn.CrossEntropyLoss().cuda()





for epoch in range(80000):

    for i, data in enumerate(train_loader, 0):
        model.train()
        iter = i + epoch * len(train_loader)
        train_loss = 0.


        batch_x,batch_y = data
        batch_y = batch_y.long()


        batch_x,batch_y= batch_x.cuda(CUDA),batch_y.cuda(CUDA)
        #print(batch_y.max() )

        out = model(batch_x)[0]
        loss = criterion(out, batch_y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_run = adjust_learning_rate(optimizer, iter)
        print("-------------------------------------------------------")
        print("Train Loss for iteration {}  is : {:.4f}  |   lr = {}  ".format(iter,train_loss, lr_run ) )


        writer.add_scalar("Train/learning_rate", lr_run, iter)
        writer.add_scalar("Train/Loss", train_loss, iter)


        
        if iter % 2000 == 1999:
            Score,class_iou = evaluate(model)
            mean_iou = Score['Mean IoU']
            writer.add_scalar("Eval/Mean_IoU", mean_iou, iter)
            torch.save(model, './check_points/res50/model_test_{}_{}.pkl'.format(epoch * len(train_loader) + i, mean_iou))

        