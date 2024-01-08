from PIL import Image
import numpy as np
import cv2

image_path = '/home/zhengm/testcode/zhufeiyu/SegFormer_CH5/SegFormer/dataset/mask-test/22082217073900_CH5_4.png'

img = cv2.imread(image_path)

# 将图像数组重构为二维数组，每行代表一个像素的BGR值
reshaped_img_array = img.reshape(-1, img.shape[-1])

# 找出数组中的唯一颜色值及其出现的次数
unique, counts = np.unique(reshaped_img_array, axis=0, return_counts=True)
pixel_counts = dict(zip(map(tuple, unique), counts))

print(pixel_counts)