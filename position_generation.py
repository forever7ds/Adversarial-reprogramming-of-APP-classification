import matplotlib.pyplot as plt
from torchvision.transforms import *
import numpy as np
from PIL import Image
picname = './data/source2/pic.jpg'   #加载图片名称
svnameposi = './data/source2/placement'
svnameblur = './data/source2/mask'  #策略,379*2,存储位置信息
from pylab import *
NUM = 379
IMG_SIZE = (224, 224)
inter = 4


def squ(arr, length, a, b):
    mea = 0
    num = 0
    for m in range(-length, length+1):
        for n in range(-length, length+1):
            if a+m<0 or b+n<0 or a+m>(IMG_SIZE[0]-1) or b+n>(IMG_SIZE[1]-1):
                continue
            mea += arr[a+m][b+n]
            num += 1
    mea /= num
    # print(num)
    out = 0
    for m in range(-length, length+1):
        for n in range(-length, length+1):
            if a+m<0 or b+n<0 or a+m>(IMG_SIZE[0]-1) or b+n>(IMG_SIZE[1]-1):
                continue
            out += (arr[a+m][b+n] - mea)**2
    out /= num
    return out


def create_blur_region():
    source = Image.open(picname).convert('RGB')
    trans = Compose([Resize(IMG_SIZE), ToTensor()])
    img_tensor = trans(source)
    img = img_tensor.numpy()
    var_matrix = np.zeros(img.shape, dtype='float32')
    blur_factor_matrix = np.zeros(img.shape, dtype='float32')
    eyer = 2

    for k in range(0, 3):
        for i in range(0, IMG_SIZE[0]):
            for j in range(0, IMG_SIZE[1]):
                var_matrix[k][i][j] = squ(img[k], eyer, i, j)

    sorted_posi = []
    for i in range(0, IMG_SIZE[0]):
        for j in range(0, IMG_SIZE[1]):
            sorted_posi.append((i, j, max([var_matrix[0][i][j], var_matrix[1][i][j], var_matrix[2][i][j]])))

    def pkey_get(item):
        return -item[2]


    def Contact(select_posi, g, r):
        cal = 0
        for num1 in range(0, len(select_posi)):
            cal += 1
            for num2 in range(max(0, select_posi[num1][0] - r), min(IMG_SIZE[0], select_posi[num1][0] + r + 1)):
                for num3 in range(max(0, select_posi[num1][1] - r), min(IMG_SIZE[1], select_posi[num1][1] + r + 1)):
                    if num2 == g[0] and num3 == g[1]:
                        return 1

        if cal == len(select_posi):
            return 0
        return -1

    sorted_posi.sort(key=pkey_get)
    select_posi = []

    for i in range(0, len(sorted_posi)):
        #print(i)
        if len(select_posi) == 0:
            select_posi.append(sorted_posi[i])
        elif len(select_posi) < NUM and Contact(select_posi, sorted_posi[i], inter) == 0:
            select_posi.append(sorted_posi[i])
        elif len(select_posi) == NUM:
            break
    # print('-----------', len(select_posi))

    for posi in select_posi:
        for c in range(0, 3):
            blur_factor_matrix[c][posi[0]][posi[1]] = 1

    for R in range(0, 3):
        maxvar = 0.0
        for posi in select_posi:
            for i in range(max(0, posi[0] - R), min(posi[0] + R + 1, IMG_SIZE[0])):
                for j in range(max(0, posi[1] - R), min(posi[1] + R + 1, IMG_SIZE[1])):
                    for k in range(0, 3):
                        if var_matrix[k][i][j] > maxvar:
                            maxvar = var_matrix[k][i][j]
        for posi in select_posi:
            for i in range(max(0, posi[0]-R), min(posi[0]+R+1, IMG_SIZE[0])):
                for j in range(max(0, posi[1]-R), min(posi[1]+R+1, IMG_SIZE[1])):
                    for k in range(0, 3):
                        if blur_factor_matrix[k][i][j] == 1:
                            continue
                        else:
                            blur_factor_matrix[k][i][j] = var_matrix[k][i][j] / maxvar
        # visual
        '''plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(img.transpose([1, 2, 0]))
        plt.subplot(2, 2, 2)
        plt.imshow(blur_factor_matrix.transpose([1, 2, 0]))
        plt.subplot(2, 2, 3)
        plt.imshow(0.5*(np.tanh(blur_factor_matrix + img*2 - 1)+1).transpose([1, 2, 0]))
        plt.show()'''
        np.save(svnameblur + 'R' + str(R), blur_factor_matrix)

    sv_posi = np.zeros(shape=(NUM, 2),dtype='int32')
    for idx in range(NUM):
        sv_posi[idx][0] = select_posi[idx][0]
        sv_posi[idx][1] = select_posi[idx][1]
    np.save(svnameposi,sv_posi)
    return


if __name__ == '__main__':
    create_blur_region()
