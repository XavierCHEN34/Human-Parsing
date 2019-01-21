import numpy as np
import cv2

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype = np.uint8)
    for i in range(N):
        r = 0
        g = 0
        b = 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ ( np.uint8(str_id[-1]) << (7-j))
            g = g ^ ( np.uint8(str_id[-2]) << (7-j))
            b = b ^ ( np.uint8(str_id[-3]) << (7-j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    return cmap



def label2color(cv2_gt, num = 2 ):
    cmap = labelcolormap(num)
    [rows, cols, _] = cv2_gt.shape
    for i in range(rows):
        for j in range(cols):
            label = cv2_gt[i,j,0]
            cv2_gt[i,j] = cmap[label]
    return cv2_gt

if __name__ == '__main__':
    big_block = np.ones((100, 100, 3))
    for i in range(1,20):
        block = np.ones((100, 100, 3)) * i
        big_block = np.concatenate((big_block, block), axis = 1 )

    mattix  =  np.zeros((200,1000,3))
    mattix[:100,:,:] = big_block[:,:1000,:]
    mattix[100:, :, :] = big_block[:, 1000:, :]
    mattix = mattix.astype(int)


    print(big_block.shape,big_block.max())

    color_map = label2color(mattix,20)
    cv2.imwrite('colormap.png', color_map)





