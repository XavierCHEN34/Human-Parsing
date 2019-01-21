import torch.nn as nn
import torch.nn.functional as F
from model.Boudarynet import BoundaryNet
import torch
from tqdm import *
import time
from config import *

class Matting_loss(nn.Module):
    def __init__(self):
        super(Matting_loss, self).__init__()
        self.Bnet = BoundaryNet()

    def forward(self, Batch_img, Batch_output, Batch_trimap ):
        Batch_trimap = Batch_trimap.float()
        Batch_img = torch.mean(Batch_img, dim=1, keepdim=True)
        img_mag, img_x, img_y = self.Bnet(Batch_img)
        img_mag[img_mag < 0.1] = 0
        o_mag, o_x, o_y = self.Bnet(Batch_output)
        #print(img_mag.max(), o_mag.max() )

        Mask = Batch_trimap
        delta_magnitude = torch.clamp( torch.clamp(img_mag , max = img_mag.max()  ) - o_mag, min = 0)
        cosine_distance = (1  - torch.abs( img_x * o_x + img_y* o_y) ) * o_mag
        weighted_distance = (cosine_distance + delta_magnitude)  * Mask

        pix = torch.zeros(BATCH_SIZE,1, 256, 256).cuda(CUDA)
        pix[img_mag > 0.1] = 1
        pix = pix * Mask
        sum_mag = pix.sum()
        return  torch.sum(weighted_distance) / sum_mag




if __name__ == '__main__':
    Batch_img = torch.rand(8,3,512,512).cuda()
    Batch_out = torch.rand(8,2,512,512).cuda()
    Batch_trimap = torch.rand(8, 512, 512).cuda()
    cri = Matting_loss().cuda()
    t1 = time.time()

    for i in tqdm( range(1)):
        loss = cri( Batch_img, Batch_out ,Batch_trimap)
    t2 = time.time()
    print(loss.item())

    print(1000/(t2-t1) )