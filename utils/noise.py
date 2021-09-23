import numpy as np
import cv2  # opencv库
import os
import glob


def read_image(img_path):
    """
    读取图片，图片是以 np.array 类型存储
    :param img_path: 图片的路径以及名称
    :return: img np.array 类型存储
    """
    # 读取图片
    img = cv2.imread(img_path)

    # 如果图片是三通道，采用 matplotlib 展示图像时需要先转换通道
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def normalization(image):
    """
    将数据线性归一化
    :param image: 图片矩阵，一般是np.array 类型 
    :return: 将归一化后的数据，在（0,1）之间
    """
    # 获取图片数据类型对象的最大值和最小值
    info = np.iinfo(image.dtype)
    
    # 图像数组数据放缩在 0-1 之间
    return image.astype(np.double) / info.max


def save_image(filename, image):
    """
    将np.ndarray 图像矩阵保存为一张 png 或 jpg 等格式的图片
    :param filename: 图片保存路径及图片名称和格式
    :param image: 图像矩阵，一般为np.array
    :return:
    """
    # np.copy() 函数创建一个副本。
    # 对副本数据进行修改，不会影响到原始数据，它们物理内存不在同一位置。
    img = np.copy(image)

    # 从给定数组的形状中删除一维的条目
    img = img.squeeze()

    # 将图片数据存储类型改为 np.uint8
    if img.dtype == np.double:
        # 若img数据存储类型是 np.double ,则转化为 np.uint8 形式
        img = img * np.iinfo(np.uint8).max

        # 转换图片数组数据类型
        img = img.astype(np.uint8)

    # 将 RGB 方式转换为 BGR 方式
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # 生成图片
    cv2.imwrite(filename, img)


def binary_noise_mask_image(img, noise_ratio, batch=True):
    """
    生成受损图片
    :param img: 图像矩阵，一般为 np.ndarray
    :param noise_ratio: 噪声比率，可能值是0.4/0.6/0.8
    :return: noise_img 受损图片
    """
    if batch:
        mask = np.random.choice([0, 1], size=(img.shape[0], img.shape[1], img.shape[2], img.shape[3]),
                            p=[noise_ratio, 1 - noise_ratio])
    else :
        mask = np.random.choice([0, 1], size=(img.shape[0], img.shape[1], img.shape[2]),
                            p=[noise_ratio, 1 - noise_ratio])
    noise_img = mask * img

    return noise_img


def add_binary_noise_to_img(raw_dir_name, save_dir_name, noise_ratio):
    fnames = glob.glob(os.path.join(raw_dir_name, '*'))
    for fname in fnames:
        img = read_image(fname)
        img = binary_noise_mask_image(img, noise_ratio, batch=False).astype(np.uint8)

        _, filename = os.path.split(fname)
        filename = filename[:-4]
        save_path = os.path.join(save_dir_name, filename+str(round(noise_ratio, 2))+".jpg")
        # 将 RGB 方式转换为 BGR 方式
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # 生成图片
        cv2.imwrite(save_path, img)


if __name__ == '__main__':
    raw_dir_name = "../normal_img"
    save_dir_name = "../noise_img"
    add_binary_noise_to_img(raw_dir_name, save_dir_name, 0.4)