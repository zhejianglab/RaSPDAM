############################################################################################
# 生成伪样本数据集
############################################################################################
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
import glob

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.io import imread, imsave
import cv2
from tqdm import tqdm

np.random.seed(0)
import os


# npys=glob.glob("./data/*.npy")

def norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def ft(res):
    rest = np.zeros_like(res)
    for i in range(3, 18, 2):
        kernel1 = -np.ones((i, i)) + i * np.eye(i)

        rest += np.abs(cv2.filter2D(res, -1, kernel=kernel1))
    rest = np.abs(rest)
    rest = rest.transpose(1, 0)
    rest /= np.mean(rest, axis=0)
    rest = rest.transpose(1, 0)
    rest = norm(rest)
    return rest


# 储存FRB信号曲线的斜率
secant = [1.28125, 1.0625, 0.90625, 0.765625, 0.65625, 0.578125, 0.5, 0.45]


def ft_set(res):
    rest = np.zeros_like(res)

    cut = 30
    k_set = secant[::-1]

    x_p = np.floor(np.linspace(0, cut - 1, cut)).astype(np.int32)
    x_n = -x_p[::-1]

    split = 0
    tem = 512 // len(k_set)

    for k in k_set:

        km = k if k <= 1.0 else 1 / k

        y_p = np.floor(km * x_p).astype(np.int32)
        y_n = -y_p[::-1]

        for ker_size in range(3, 18, 2):
            ker = np.zeros((ker_size, ker_size))
            num = ker_size // 2

            x_n_tem, x_p_tem = x_n[-num - 1:] + num, x_p[:num + 1] + num
            y_n_tem, y_p_tem = y_n[-num - 1:] + num, y_p[:num + 1] + num

            ker[x_n_tem, y_n_tem] = 1.0
            ker[x_p_tem, y_p_tem] = 1.0

            f0_num = np.sum(ker == 0)
            f1_num = np.sum(ker == 1)

            ker[ker == 0] = -f1_num / f0_num

            if k > 1.0: ker = ker.transpose(1, 0)

            rest[split * tem:(split + 1) * tem] += np.abs(cv2.filter2D(res, -1, kernel=ker))[
                                                   split * tem:(split + 1) * tem]

        split += 1

    rest = rest.transpose(1, 0)
    mean_val = np.mean(rest, axis=0)
    mean_val[mean_val == 0.0] = 1.0
    rest /= mean_val
    rest = rest.transpose(1, 0)
    rest = norm(rest)

    return rest


def ns_gen():
    # 条纹噪声
    cut = 512
    img_noise = np.random.random((cut, 1))
    mask = np.where(np.random.random((cut, 1)) < 0.1, 1.0, 0.0)

    mask_mask = np.linspace(0.0, 1.0, cut // 3)
    mask[:len(mask_mask), 0] *= mask_mask
    mask[-len(mask_mask):, 0] *= mask_mask[::-1]

    kernel1 = np.array([[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]])
    img_noise = cv2.filter2D(img_noise, -1, kernel=kernel1)
    mask = cv2.filter2D(mask, -1, kernel=kernel1)

    img_noise -= np.mean(img_noise)

    mat = np.linspace(img_noise, -img_noise, cut)[:, :, 0].transpose(1, 0)

    return mask * mat


def flat(res):
    flat_data = np.sort(res.flatten())
    tshd = 1
    vmin, vmax = flat_data[50 * len(flat_data) // 100], flat_data[(100 - tshd) * len(flat_data) // 100]
    res = np.clip(res, vmin, vmax)
    res = norm(res)
    return res


def noise_gen():
    # 白噪声
    cut = 512
    img_noise = np.random.random((cut, cut))
    img_noise[img_noise < np.random.random() * 0.5] = 0
    return img_noise


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def frb_gen():
    cut = 512  # 图像尺寸
    num = 4096  # 频率通道数

    # 生成FRB信号曲线的斜率，并储存至secant
    #    global secant
    #    cut_num = 8
    #    cut_size = len(freq_norm) // cut_num
    #    for img_cut in range(cut_num):
    #        dy = y_norm[cut_size * (img_cut + 1) - 1] - y_norm[cut_size * img_cut]
    #        dx = freq_norm[cut_size * (img_cut + 1) - 1] - freq_norm[cut_size * img_cut]
    #        secant.append(dy / dx)

    i = 0
    tbar = tqdm(range(3333))  # 3333*3=9999  生成样本数量
    for xff in tbar:

        dm = 0
        while dm < 50: dm = 300 + 150 * np.random.randn()  # 生成dm值

        freq = np.linspace(1000, 1500, num)  # 频率 1~1.5 GHz
        y = (4.15 * dm * (freq ** -2 - freq.max() ** -2) * 1e3 / (49.152 / 1e6)).astype(
            np.int64)  # 根据 FRB特征表达式，生成一系列信号点
        y_norm = ((0.8 * y / 20000 + 0.1) * 511).astype(np.int64)  # 移动至图像中央
        freq_norm = (norm(freq) * 511).astype(np.int64)[::-1]
        y_norm = np.clip(y_norm, 0, 511)
        freq_norm = np.clip(freq_norm, 0, 511)

        img = np.zeros((cut, cut))  # 空图
        mask = np.random.random((num)) > 0.5  # 随机掩码
        img[freq_norm[mask], y_norm[mask]] = 1.0  # 保留50%的信号点

        k, b = np.random.random(), np.random.random() * cut
        while k == 0.0: k, b = np.random.random(), np.random.random() * cut

        fd = norm(np.clip(np.sin(k * 0.2 * np.linspace(0, cut - 1, cut) + b), 0, 1))  # 利用高频sin函数，模拟FRB信号断断续续的结果
        fd0 = norm(np.clip(np.sin(k * 0.02 * np.linspace(0, cut - 1, cut)), 0, 1))[::-1]  # 利用低频sin函数，模拟FRB信号断断续续的结果

        # 三种样本
        img0 = img * fd0  # 低频sin函数 间隔较大的断断续续
        img1 = img * fd  # 高频sin函数 间隔较大的断断续续
        img2 = img * fd0 * fd  # 相乘 难例样本（FRB弱信号）

        # 三种标签GT
        lab0 = np.where(img0 > 0.25, 1.0, 0.0)
        lab1 = np.where(img1 > 0.25, 1.0, 0.0)
        lab2 = np.where(img2 > 0.25, 1.0, 0.0)

        img_set = [img0, img1, img2]
        lab_set = [lab0, lab1, lab2]
        kernel = np.ones((3, 3), np.uint8)

        # 对线条进行形态学膨胀处理，模拟FRB线条的粗细变化
        for tem_res in range(len(lab_set)):
            for n in range(1, 10):
                if np.random.random() < 0.8 * 0.5 ** n:
                    img_set[tem_res] = cv2.dilate(img_set[tem_res], kernel, iterations=1)
                    lab_set[tem_res] = cv2.dilate(lab_set[tem_res], kernel, iterations=1)

            lab_set[tem_res] = cv2.dilate(lab_set[tem_res], kernel, iterations=1)

        img0, img1, img2 = img_set

        # 白噪声 条纹噪声 FRB信号 三种信号的耦合
        res1 = (1.5 * noise_gen() + 10 * ns_gen() + 1 * img0) ** 2
        res2 = (1.5 * noise_gen() + 10 * ns_gen() + 1 * img1) ** 2
        res3 = (1.5 * noise_gen() + 10 * ns_gen() + 1 * img2) ** 2

        path_all = ["./Lab1-Y/", "./Tem1-Y1/", "./Tem1-Y2/", "./Tem1-Y3/"]
        for path in path_all:
            check_dir(path)

        for tem_res in range(len(lab_set)):
            cv2.imwrite(path_all[0] + "%05d.png" % i, lab_set[tem_res] * 255)
            i += 1

        res_set = [flat(res1), flat(res2), flat(res3)]
        for path in path_all[1:]:
            i -= len(res_set)
            for tem_res in range(len(res_set)):
                if "Y1" in path:
                    pass

                elif "Y2" in path:
                    res_set[tem_res] = ft_set(res_set[tem_res])  # 斜线检测

                elif "Y3" in path:
                    res_set[tem_res] = cv2.dilate(res_set[tem_res], kernel, iterations=1)  # 斜线检测结果的形态学膨胀

                cv2.imwrite(path + "%05d.png" % i, res_set[tem_res] * 255)
                i += 1


frb_gen()
