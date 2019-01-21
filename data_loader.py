import cv2
import numpy as np
np.set_printoptions(threshold=np.nan)
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import ConcatDataset
from PIL import Image
import os
from config import *
from utils.data_aug_online_P  import *
import torch
np.random.seed(1)




normalize = transforms.Normalize(
    #mean=[0.485, 0.456, 0.406],
    mean = [0.51052445,0.45063994,0.41973213],
    #std=[0.229, 0.224, 0.225]
    std = [0.26807685,0.25218709,0.2435979 ]
)


preprocess_img = transforms.Compose([
    transforms.ToTensor(),
    normalize
])


def preprocess_gt(PIL_bd):
    np_bd = np.array(PIL_bd)

    if np_bd.shape[-1] == 3:
        np_bd = np_bd[:,:,0]
    tensor_bd = torch.from_numpy(np_bd)
    return tensor_bd




def data_aug(img,gt):
    #img,gt = Random_Size_P(img,gt)
    #img,gt = Random_Move_P(img,gt)
    img,gt = Random_Rotation_P(img,gt)
    img,gt= Random_flip_P(img,gt)
    img,gt = Random_light_P(img,gt)
    #img,gt = Random_blur_P(img,gt)
    return img,gt



def train_loader(PIL_imag,PIL_gt):
    img_tensor = preprocess_img(PIL_imag)
    gt_tensor = preprocess_gt(PIL_gt)
    return img_tensor,gt_tensor


class LIP(Dataset):
    def __init__(self, loader=train_loader):
        self.ads_img= train_img_lst
        self.loader = loader
        self.data_aug = data_aug

    def __getitem__(self, index):
        ad_img = dir_train_imgs +  self.ads_img[index]
        ad_gt = dir_train_gt + self.ads_img[index].replace('jpg','png')
        PIL_img, PIL_gt = Image.open(ad_img),Image.open(ad_gt)
        PIL_img = PIL_img.convert('RGB')


        img = PIL_img.resize( (128,256), Image.NEAREST )
        gt = PIL_gt.resize((128, 256), Image.NEAREST)


        
        img,gt= self.data_aug(img,gt)
        img_tensor,gt_tensor = self.loader(img,gt)
        return img_tensor,gt_tensor#[:,:,0]#.unsqueeze_(0)
    def __len__(self):
        return len(self.ads_img)






if  __name__ == '__main__':
    set =LIP()
    data_loader = DataLoader(set, batch_size=1,shuffle=True,num_workers=8)
    print(len(data_loader))
    for (batch_x, batch_y) in data_loader:
            print(batch_x.size(),batch_y.size())
            print(np.unique(batch_y) )
