'''
File        : compare.py
Date        : 2024/1/8
Author      : Feiyu Zhu
Detail      : 选择向量数据
'''
from skimage import io, transform
from skimage.metrics import structural_similarity as ssim
import numpy as np
import os
import pandas as pd

def calculate_image_similarity(image1, image2):
    # 确保两个图像的尺寸相同
    if image1.shape != image2.shape:
        # 将image2调整为与image1相同的尺寸
        image2 = transform.resize(image2, image1.shape)

    # 确定data_range
    if image1.dtype == np.float32 or image1.dtype == np.float64:
        data_range = 1.0
    else:
        data_range = 255

    # 计算SSIM并指定data_range
    similarity_index = ssim(image1, image2, data_range=data_range)

    return similarity_index

def calculate_folder_similarity(folder_path):
    # 获取文件夹中所有图片的路径
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.bmp', '.jpg', '.jpeg'))]
    num_images = len(image_paths)

    # 创建一个空矩阵来存储相似度值
    similarity_matrix = np.zeros((num_images, num_images))

    # 计算每一对图片的相似度
    for i in range(num_images):
        for j in range(num_images):
            if i != j:
                image1 = io.imread(image_paths[i], as_gray=True)
                image2 = io.imread(image_paths[j], as_gray=True)
                similarity_matrix[i, j] = calculate_image_similarity(image1, image2)
            else:
                similarity_matrix[i, j] = 1  # 同一张图片的相似度为1

    # 获取文件名作为标签
    labels = [os.path.basename(path) for path in image_paths]
    return similarity_matrix, labels

def save_similarity_matrix_to_excel(matrix, labels, folder_path):
    # 创建DataFrame
    df = pd.DataFrame(matrix, index=labels, columns=labels)

    # 生成Excel文件名
    excel_filename = os.path.join(folder_path, 'similarity_matrix.xlsx')

    # 保存到Excel
    df.to_excel(excel_filename)
    print(f"相似度矩阵已保存至: {excel_filename}")


if __name__ == '__main__':
    # 文件夹路径
    folder_path = '/home/zhengm/testcode/zhufeiyu/Painter/SegGPT/SegGPT_inference/cimai_data/label/image'
    similarity_matrix, labels = calculate_folder_similarity(folder_path)
    print(similarity_matrix)
    save_similarity_matrix_to_excel(similarity_matrix, labels, folder_path)
