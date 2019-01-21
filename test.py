import cv2
from utils.eval_IOU import *
import torch
from model.deeplabv2 import deeplab_res50 ,deeplab_res101
from model.resnet_tiny_refine import resnet_tiny_refine
from model.EDAnet import EDANet
from config import *


model = deeplab_res50(20).cuda()
model.load_state_dict(torch.load( 'deeplav_resnet50_c20.pkl' ))
model.eval()
#torch.save(model.state_dict(), 'deeplab_101.pkl')
#evaluate(model, overlap=True, merge_channel=True)
test_dir(model, '/home/cx/Dataset/reid/Market-1501-v15.09.15/bounding_box_test/',merge_channel=True )