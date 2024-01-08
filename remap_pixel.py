'''
File        : remap_pixel.py
Date        : 2024/1/8
Author      : Feiyu Zhu
Detail      : 像素转化
'''

from PIL import Image
import numpy as np
import os

def remap_pixel_values(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):  # 假设掩码文件是PNG格式
            file_path = os.path.join(input_folder, filename)
            img = Image.open(file_path)
            img_array = np.array(img)

            # 假设原图像是单通道，并且像素值为类别标签
            # 创建一个新的三通道数组，用于保存映射后的颜色
            color_mapped_img = np.zeros((img_array.shape[0], img_array.shape[1], 3), dtype=np.uint8)

            # 映射颜色
            color_mapped_img[img_array == 1] = [255, 255, 255]  # 白色
            color_mapped_img[img_array == 2] = [255, 0, 0]  # 红色
            color_mapped_img[img_array == 3] = [0, 255, 0]  # 黄色

            # 将NumPy数组转回PIL图像并保存
            new_img = Image.fromarray(color_mapped_img)
            new_file_path = os.path.join(output_folder, filename)
            new_img.save(new_file_path)

def remap_color_to_gray(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            file_path = os.path.join(input_folder, filename)
            img = Image.open(file_path)
            img_array = np.array(img)

            # 创建一个新的单通道数组，用于保存灰度值
            gray_img = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)

            # 逆映射颜色
            # 假设白色[255, 255, 255]映射回1，红色[255, 0, 0]映射回2，绿色[0, 255, 0]映射回3
            gray_img[np.all(img_array == [255, 255, 255], axis=-1)] = 1
            gray_img[np.all(img_array == [255, 0, 0], axis=-1)] = 2
            gray_img[np.all(img_array == [0, 255, 0], axis=-1)] = 3

            # 将NumPy数组转回PIL图像并保存
            new_img = Image.fromarray(gray_img)
            new_file_path = os.path.join(output_folder, filename)
            new_img.save(new_file_path)


# 使用示例
input_folder = '/home/zhengm/testcode/zhufeiyu/SegFormer_CH5/SegFormer/dataset/mask-tset'  # 替换为您的输入文件夹路径
output_folder = '/home/zhengm/testcode/zhufeiyu/SegFormer_CH5/SegFormer/dataset/mask-test'  # 替换为您的输出文件夹路径
remap_pixel_values(input_folder, output_folder)
