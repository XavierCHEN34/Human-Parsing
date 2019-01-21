from PIL import Image,ImageFilter,ImageEnhance
import numpy as np

def Random_Rotation(PIL_img,PIL_gt,PIL_bd):
    possible = np.random.randint(2)
    if possible == 0:
        return PIL_img,PIL_gt,PIL_bd

    degree = np.random.randint(-45,46)  #45-30
    img_rot = PIL_img.rotate(degree)
    gt_rot = PIL_gt.rotate(degree)
    bd_rot = PIL_bd.rotate(degree)

    return img_rot,gt_rot,bd_rot


def Random_flip(PIL_img,PIL_gt,PIL_bd):
    possible = np.random.randint(2)
    if possible == 0:
        return PIL_img,PIL_gt,PIL_bd
    img_fp = PIL_img.transpose(Image.FLIP_LEFT_RIGHT)
    gt_fp = PIL_gt.transpose(Image.FLIP_LEFT_RIGHT)
    bd_fp = PIL_bd.transpose(Image.FLIP_LEFT_RIGHT)

    return img_fp,gt_fp,bd_fp

def Random_color(PIL_img,PIL_gt):
    possible = np.random.randint(2)
    if possible == 0:
        return PIL_img, PIL_gt


    color = np.random.randint(6,10)/10    #5,16 -- 7,13
    img_lt = ImageEnhance.Color(PIL_img).enhance(color)
    return img_lt, PIL_gt

def Random_constract(PIL_img,PIL_gt):
    possible = np.random.randint(2)
    if possible == 0:
        return PIL_img, PIL_gt


    con = np.random.randint(6,10)/10    #5,16 -- 7,13
    img_lt = ImageEnhance.Contrast(PIL_img).enhance(con)
    return img_lt, PIL_gt


class MyGaussianBlur(ImageFilter.Filter):
    name = "GaussianBlur"

    def __init__(self, radius=2, bounds=None):
        self.radius = radius
        self.bounds = bounds

    def filter(self, image):
        if self.bounds:
            clips = image.crop(self.bounds).gaussian_blur(self.radius)
            image.paste(clips, self.bounds)
            return image
        else:
            return image.gaussian_blur(self.radius)

def Random_blur(PIL_img,PIL_gt,PIL_bd):
    possible = np.random.randint(2)
    if possible == 0:
        return PIL_img, PIL_gt,PIL_bd
    rad = np.random.randint(5)     #8-5
    img_bl = PIL_img.filter(MyGaussianBlur(radius=rad))
    return img_bl, PIL_gt,PIL_bd


def Random_Size(PIL_img,PIL_gt,PIL_bd):
    possible = np.random.randint(2)
    if possible == 0:
        return PIL_img, PIL_gt, PIL_bd
    w, h = PIL_img.size
    r = np.random.randint(4)/10
    img_sz = PIL_img.crop((- w * r, - h *r , w *(1 + r) , h *(1 + r) )).resize((256,256))
    gt_sz = PIL_gt.crop((- w * r, - h * r, w * (1 + r), h * (1 + r))).resize((256,256))
    bd_sz = PIL_bd.crop((- w * r, - h * r, w * (1 + r), h * (1 + r))).resize((256, 256))
    return img_sz, gt_sz, bd_sz

def Random_Move(PIL_img,PIL_gt,PIL_bd):
    possible = np.random.randint(2)
    if possible == 0:
        return PIL_img, PIL_gt,PIL_bd
    w, h = PIL_img.size
    rx = np.random.randint(-2, 3)/10
    ry = np.random.randint(-2, 3) / 10
    img_mv = PIL_img.crop((- w * rx, - h *ry , w*(1-rx) , h*(1-ry)  ))
    gt_mv = PIL_gt.crop((- w * rx, - h *ry , w*(1-rx) , h*(1-ry)  ))
    bd_mv = PIL_bd.crop((- w * rx, - h * ry, w * (1 - rx), h * (1 - ry)))
    return img_mv, gt_mv,bd_mv


def Random_light(PIL_img,PIL_gt,PIL_bd):
    possible = np.random.randint(2)
    if possible == 0:
        return PIL_img, PIL_gt,PIL_bd


    Gamma = np.random.randint(7,13)/10    #5,16 -- 7,13
    img_lt = ImageEnhance.Brightness(PIL_img).enhance(Gamma)
    return img_lt, PIL_gt,PIL_bd