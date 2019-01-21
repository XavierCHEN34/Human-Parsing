from PIL import Image,ImageFilter,ImageEnhance
import numpy as np

def Random_Rotation_P(PIL_img,PIL_gt):
    possible = np.random.randint(2)
    if possible == 0:
        return PIL_img,PIL_gt

    degree = np.random.randint(-90,90)  #45-30
    img_rot = PIL_img.rotate(degree)
    gt_rot = PIL_gt.rotate(degree)

    return img_rot,gt_rot


def Random_flip_P(PIL_img,PIL_gt):
    possible = np.random.randint(2)
    if possible == 0:
        return PIL_img,PIL_gt
    img_fp = PIL_img.transpose(Image.FLIP_LEFT_RIGHT)
    gt_fp = PIL_gt.transpose(Image.FLIP_LEFT_RIGHT)

    return img_fp,gt_fp

def Random_color_P(PIL_img,PIL_gt):
    possible = np.random.randint(2)
    if possible == 0:
        return PIL_img, PIL_gt


    color = np.random.randint(6,10)/10    #5,16 -- 7,13
    img_lt = ImageEnhance.Color(PIL_img).enhance(color)
    return img_lt, PIL_gt

def Random_constract_P(PIL_img,PIL_gt):
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

def Random_blur_P(PIL_img,PIL_gt):
    possible = np.random.randint(2)
    if possible == 0:
        return PIL_img, PIL_gt
    rad = np.random.randint(5)     #8-5
    img_bl = PIL_img.filter(MyGaussianBlur(radius=rad))
    return img_bl, PIL_gt


def Random_Size_P(PIL_img,PIL_gt):
    possible = np.random.randint(2)
    if possible == 0:
        return PIL_img, PIL_gt
    w, h = PIL_img.size
    r = np.random.randint(5)/10
    img_sz = PIL_img.crop((- w * r, - h *r , w *(1 + r) , h *(1 + r) )).resize((320,416))
    gt_sz = PIL_gt.crop((- w * r, - h * r, w * (1 + r), h * (1 + r))).resize((320,416))
    return img_sz, gt_sz

def Random_Move_P(PIL_img,PIL_gt):
    possible = np.random.randint(2)
    if possible == 0:
        return PIL_img, PIL_gt
    w, h = PIL_img.size
    rx = np.random.randint(-2, 3)/10
    ry = np.random.randint(-2, 3) / 10
    img_mv = PIL_img.crop((- w * rx, - h *ry , w*(1-rx) , h*(1-ry)  ))
    gt_mv = PIL_gt.crop((- w * rx, - h *ry , w*(1-rx) , h*(1-ry)  ))
    return img_mv, gt_mv


def Random_light_P(PIL_img,PIL_gt):
    possible = np.random.randint(2)
    if possible == 0:
        return PIL_img, PIL_gt


    Gamma = np.random.randint(7,13)/10    #5,16 -- 7,13
    img_lt = ImageEnhance.Brightness(PIL_img).enhance(Gamma)
    return img_lt, PIL_gt