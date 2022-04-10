import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

dict = [[0,0,0],[128,0,0],[0,128,0],[128,128,0],
        [0,0,128],[0,128,128],[128,128,128],[192,0,0],
        [64,128,0],[192,128,0],[64,0,128],[192,0,128],
        [64,128,128],[192,128,128],[0,64,0],[128,64,0],
        [0,192,0],[128,64,128],[0,192,128],[128,192,128],
        [64,64,0],[192,64,0]]
#dict = np.array(dict, dtype=np.uint8)
path_img = r'D:\pythonwork\深度学习\第二次\msrc2_seg\images'
path_gt = r'D:\pythonwork\深度学习\第二次\msrc2_seg\gt'


def processing(img, mask):
    img = cv2.imdecode(np.fromfile(os.path.join(path_img, img), dtype=np.uint8), 1)[:, :, [2, 1, 0]]
    mask = cv2.imdecode(np.fromfile(os.path.join(path_gt, mask), dtype=np.uint8), 1)[:, :, [2, 1, 0]]
    (h, w, channels) = img.shape
    if(h > w and w < 240):
        w1 = (240-w)//2
        w2 = 240 - w1 - w
        img = cv2.copyMakeBorder(img, 0, 0, w1, w2, cv2.BORDER_REFLECT_101)
        mask = cv2.copyMakeBorder(mask, 0, 0, w1, w2, cv2.BORDER_REFLECT_101)
    elif(h < w ):
        if(h < 240):
            h1 = (240 - h) // 2
            h2 = 240 - h1 - h
            img = cv2.copyMakeBorder(img, h1, h2, 0, 0, cv2.BORDER_REFLECT_101)
            mask = cv2.copyMakeBorder(mask, h1, h2, 0, 0, cv2.BORDER_REFLECT_101)
        img = [img[:, :, 0].T.reshape(320, 240, 1), img[:, :, 1].T.reshape(320, 240, 1), img[:, :, 2].T.reshape(320, 240, 1)]
        img = np.concatenate(img, axis=2)
        mask = [mask[:,:,0].T.reshape(320, 240,1), mask[:,:,1].T.reshape(320, 240,1), mask[:,:,2].T.reshape(320, 240,1)]
        mask = np.concatenate(mask, axis=2)

    #print(mask.shape)
    map = np.zeros((320, 240), dtype=np.uint8)
    for i in range(320):
        for j in range(240):
            index = dict.index(list(mask[i, j, :]))
            map[i, j] = index
    '''
    outputs = np.zeros(img.shape, dtype=np.uint8)
    for i in range(len(dict)):
        index = map==i
        outputs[index] = dict[i]
    plt.subplot(151)
    plt.imshow(img)
    plt.subplot(152)
    plt.imshow(mask)
    plt.subplot(153)
    plt.imshow(map)
    plt.subplot(154)
    plt.imshow(outputs)
    plt.subplot(155)
    plt.imshow(mask-outputs)
    plt.show()
    #print(np.sort(map.reshape(-1,)))'''
    return (img, map)

def traingenertor(train_imgs):
    i = 0
    while 1:
        i = (i+1)//len(train_imgs)
        inputs = []
        outputs = []
        while( i//3 !=0):
            img = train_imgs[i]
            split = img.split('.')
            mask = split[0] + '_GT.' + split[1]
            inp, out = processing(img, mask)
            inputs.append(inp)
            outputs.append(out)
            i = (i+1)//len(train_imgs)
        img = train_imgs[i]
        split = img.split('.')
        mask = split[0] + '_GT.' + split[1]
        inp, out = processing(img, mask)
        inputs.append(inp)
        outputs.append(out)
        yield np.array(inputs), np.array(outputs, dtype=np.uint8)


def show_image(img, model):
    split = img.split('.')
    mask = split[0] + '_GT.' + split[1]
    img = cv2.imdecode(np.fromfile(os.path.join(path_img, img), dtype=np.uint8), 1)[:, :, [2, 1, 0]]
    mask = cv2.imdecode(np.fromfile(os.path.join(path_gt, mask), dtype=np.uint8), 1)[:, :, [2, 1, 0]]
    (h, w, channels) = img.shape
    if (h > w and w < 240):
        w1 = (240 - w) // 2
        w2 = 240 - w1 - w
        img = cv2.copyMakeBorder(img, 0, 0, w1, w2, cv2.BORDER_REFLECT_101)
        mask = cv2.copyMakeBorder(mask, 0, 0, w1, w2, cv2.BORDER_REFLECT_101)
    elif (h < w):
        if(h < 240):
            h1 = (240 - h) // 2
            h2 = 240 - h1 - h
            img = cv2.copyMakeBorder(img, h1, h2, 0, 0, cv2.BORDER_REFLECT_101)
            mask = cv2.copyMakeBorder(mask, h1, h2, 0, 0, cv2.BORDER_REFLECT_101)
        img = [img[:, :, 0].T.reshape(320, 240, 1), img[:, :, 1].T.reshape(320, 240, 1),
               img[:, :, 2].T.reshape(320, 240, 1)]
        img = np.concatenate(img, axis=2)
        mask = [mask[:, :, 0].T.reshape(320, 240, 1), mask[:, :, 1].T.reshape(320, 240, 1),
                mask[:, :, 2].T.reshape(320, 240, 1)]
        mask = np.concatenate(mask, axis=2)

    img = np.array([img])
    mask = np.array([mask])
    y1 = model.predict(img)[0]
    y = np.argmax(y1, axis=2)
    output = np.zeros(img[0].shape, dtype=np.uint8)
    for i in range(len(dict)):
        index = y==i
        output[index] = dict[i]
    print(y1[0,0,:])
    print(np.sum(y1[:,0,0]))
    plt.subplot(141)
    plt.imshow(img[0])
    plt.subplot(142)
    plt.imshow(mask[0])
    plt.subplot(143)
    plt.imshow(output)
    plt.subplot(144)
    plt.imshow(y)
    plt.savefig('zuhe.png')
    plt.show()
'''
img ='1_18_s.bmp'
mask ='1_18_s_GT.bmp'
processing(img, mask)'''

