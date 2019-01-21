from config import *
from data_loader import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from tqdm import *
from config import *
import time
import random
import shutil
from utils.metric import scores
from utils.color_map import label2color
import copy
from PIL import  Image

normalize = transforms.Normalize(
    # mean=[0.485, 0.456, 0.406],
    mean=[0.51052445, 0.45063994, 0.41973213],
    # std=[0.229, 0.224, 0.225]
    std=[0.26807685, 0.25218709, 0.2435979]
)
preprocess = transforms.Compose([
    transforms.ToTensor(),
    normalize
])



def merge_channels(input):
    '''
    :param input:  (1,20,256,128)
    :return: (1,5,256,128)
    '''
    out = np.zeros((1,5,256,128))
    out[:,0,:,:] = input[:,0,:,:]
    out[:,1,:,:] = input[:, 1, :, :] + input[:, 2, :, :]+input[:, 4, :, :]+input[:, 13, :, :]
    out[:,2,:,:] = input[:, 3, :, :] + input[:, 5, :, :]+input[:, 6, :, :]+ input[:, 7, :, :]+input[:, 10, :, :]+input[:, 11, :, :]+input[:, 14, :, :]+input[:, 15, :, :]
    out[:,3,:,:] = input[:, 9, :, :] + input[:, 12, :, :]+input[:, 16, :, :]+input[:, 17, :, :]
    out[:,4,:,:] = input[:, 8, :, :] + input[:, 18, :, :]+input[:, 19, :, :]
    return out



def preprocess_gt(PIL_bd):
    np_bd = np.array(PIL_bd)
    tensor_bd = torch.from_numpy(np_bd)
    return tensor_bd


def gt_loader(path):
    gt_pil = Image.open(path).resize((128,256))
    gt_tensor = preprocess_gt(gt_pil)
    if gt_tensor.shape[-1] == 3:
        gt_tensor = gt_tensor[:, :, 0]
    return gt_tensor.unsqueeze_(0)



def default_loader(path):
    img_pil = Image.open(path).convert('RGB')
    img_pil = img_pil.resize((128, 256), Image.NEAREST)
    img_tensor = preprocess(img_pil)
    return img_tensor







def evaluate(model,overlap = True,merge_channel = True):
    shutil.rmtree('./pre')
    time.sleep(1)
    os.makedirs('./pre')

    num = len(val_img_list)
    print('numbers of test images:',num)
    targets, outputs = [], []
    for i in tqdm(range(1000)):
        ad_img = dir_val_imgs  + val_img_list[i]
        ad_gt = dir_val_gt + val_img_list[i].replace('jpg', 'png')

        ad_pre = './pre/' + val_img_list[i]
        model.eval()

        img_np =  cv2.imread(ad_img)
        img_np = cv2.resize(img_np, (128,256))


        img = default_loader(ad_img)
        img.unsqueeze_(0)
        img = img.cuda(CUDA)
        out = model(img)[0]
        output = F.softmax(out, dim=1)
        output = output.data.cpu().numpy()

        if merge_channel == True:
            output = merge_channels(output)
            num_class = 5
            ad_gt = dir_val_gt_4 + val_img_list[i].replace('jpg', 'png')
        else:
            num_class = 20
        out = np.argmax(output, axis=1)
        #print(out.shape)

        mask = np.stack((out[0], out[0], out[0]), 2)
        mask = label2color(mask, num_class)

        # print(out)
        if overlap == True:
            mask = 0.6 * mask + 0.4 * img_np
            cv2.imwrite(ad_pre, mask)

        else:
            back  = np.ones((256,128,3)) * 255
            cut = np.stack((out[0],out[0],out[0]),2)
            mask = img_np * cut + back * (1-cut)
            cv2.imwrite(ad_pre, mask)

        target = gt_loader(ad_gt).numpy()


        for o, t in zip(out, target):
            outputs.append(o)
            targets.append(t)

            
 
    score, class_iou = scores(targets, outputs, n_class=num_class)
    for k, v in score.items():
        print(k, v)
    for k, v in class_iou.items():
        print(k, v)

    return  score,class_iou






def test_dir(model, dir, overlap=True,merge_channel = True):
    shutil.rmtree('./pre')
    time.sleep(1)
    os.makedirs('./pre')
    img_list = os.listdir(dir)
    num = len(img_list)
    print('numbers of test images:', num)
    for i in tqdm(range(num)):
        ad_img = dir + img_list[i]
        ad_pre = './pre/' + img_list[i]
        model.eval()

        img_np = cv2.imread(ad_img)
        img_np = cv2.resize(img_np, (128, 256))

        img = default_loader(ad_img)
        img.unsqueeze_(0)
        img = img.cuda(CUDA)
        out = model(img)[0]
        output = F.softmax(out, dim=1)
        output = output.data.cpu().numpy()
        #print(output.shape)  (1,20,256,128)
        if merge_channel == True:
            output = merge_channels(output)
            num_class = 5
        else:
            num_class = 20

        out = np.argmax(output, axis=1)
        # print(out.shape)

        mask = np.stack((out[0], out[0], out[0]), 2)
        mask = label2color(mask, num_class)

        # print(out)
        if overlap == True:
            mask = 0.6 * mask + 0.4 * img_np
            cv2.imwrite(ad_pre, mask)

        else:
            back = np.ones((256, 128, 3)) * 255
            cut = np.stack((out[0], out[0], out[0]), 2)
            mask = img_np * cut + back * (1 - cut)
            cv2.imwrite(ad_pre, mask)


