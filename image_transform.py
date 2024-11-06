import functools
import math
from typing import List

import numpy as np
import matplotlib.pyplot as plt
# from numba import jit
from skimage.measure._regionprops import RegionProperties
import cv2
from skimage import measure, color

from params import SDParams

# 通用卷积核斜率
secant = [1.3125, 1.09375, 0.9375, 0.78125, 0.671875, 0.59375, 0.515625, 0.4603174603174603]


def draw_double(img_a, img_b, title_a="titleA", title_b="titleB"):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    ax1.imshow(img_a)
    ax1.set_title(title_a, fontsize=20)

    ax2.imshow(img_b)
    ax2.set_title(title_b, fontsize=20)

    fig.tight_layout()
    plt.show()


def draw_triple(img_a, img_b, img_c, title_a="titleA", title_b="titleB", title_c="titleC"):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(8, 4))

    ax1.imshow(img_a)
    ax1.set_title(title_a, fontsize=20)

    ax2.imshow(img_b)
    ax2.set_title(title_b, fontsize=20)

    ax3.imshow(img_c)
    ax3.set_title(title_c, fontsize=20)

    fig.tight_layout()
    plt.show()


def draw_list(img_list):
    img_num = len(img_list)
    fig, ax_list = plt.subplots(nrows=1, ncols=img_num, figsize=(8, 4))

    for i in range(len(img_list)):
        img = img_list[i]
        ax = ax_list[i]

        ax.imshow(img)
        ax.set_title("IMG: {}".format(i), fontsize=20)

    fig.tight_layout()
    plt.show()


def draw_single(img):
    plt.imshow(img)
    plt.show()


def norm(x):
    denominator = np.max(x) - np.min(x)
    if denominator == 0:
        return (x - np.min(x)) / 1
    return (x - np.min(x)) / denominator


def flat(res):
    flat_data = np.sort(res.flatten())
    tshd = 1
    vmin, vmax = flat_data[50 * len(flat_data) // 100], flat_data[(100 - tshd) * len(flat_data) // 100]
    res = np.clip(res, vmin, vmax)
    res = norm(res)
    return res


# @jit(nopython=True)
def resize_norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


# @jit(nopython=True)
def image_resize(input_image):
    """
    输入二维矩阵，压缩成512*512大小
    """

    image_data = input_image.copy()

    cut = 512  # resize的大小
    h, w = image_data.shape

    ch, cw = h // cut, w // cut
    # 多个像素点取均值，转为512*512
    image_data = np.resize(image_data, (cut, ch, cut, cw))
    # image_data.resize(cut, ch, cut, cw)
    image_rescaled = np.mean(image_data, axis=(1, 3))
    image_rescaled = resize_norm(image_rescaled)
    image_rescaled -= np.mean(image_rescaled, axis=0)
    image_rescaled = resize_norm(image_rescaled)
    res = np.rot90(image_rescaled)
    flat_data = np.sort(image_rescaled.flatten())
    tshd = 5
    vmin, vmax = flat_data[20 * len(flat_data) // 100], flat_data[(100 - tshd) * len(flat_data) // 100]
    res = np.clip(res, vmin, vmax)
    res = resize_norm(res)

    return res


def ft_set(res):
    rest = np.zeros_like(res)

    cut = 30
    k_set = secant[::-1]
    x_p = np.floor(np.linspace(0, cut - 1, cut)).astype(np.int32)  # 生成右半部分自变量
    x_n = -x_p[::-1]  # 生成左半部分自变量

    split = 0
    tem = 512 // len(k_set)

    for k in k_set:  # 生成不同斜率的卷积核

        km = k if k <= 1.0 else 1 / k

        y_p = np.floor(km * x_p).astype(np.int32)  # 根据 x坐标、斜率k --> y坐标
        y_n = -y_p[::-1]

        for ker_size in range(3, 18, 2):  # 生成不同尺寸卷积核
            ker = np.zeros((ker_size, ker_size))
            num = ker_size // 2

            x_n_tem, x_p_tem = x_n[-num - 1:] + num, x_p[:num + 1] + num
            y_n_tem, y_p_tem = y_n[-num - 1:] + num, y_p[:num + 1] + num

            ker[x_n_tem, y_n_tem] = 1.0  # 赋值卷积核左半部分直线
            ker[x_p_tem, y_p_tem] = 1.0  # 赋值卷积核右半部分直线

            f0_num = np.sum(ker == 0)
            f1_num = np.sum(ker == 1)
            ker[ker == 0] = -f1_num / f0_num  # 赋值其他区域 使得卷积核总和为0

            if k > 1.0:
                ker = ker.transpose(1, 0)

            # 切片 卷积 叠加
            rest[split * tem:(split + 1) * tem] += np.abs(
                cv2.filter2D(res[split * tem:(split + 1) * tem], -1, kernel=ker)
            )

        split += 1

    # 后处理
    rest = rest.transpose(1, 0)
    mean_val = np.mean(rest, axis=0)
    mean_val[mean_val == 0.0] = 1.0
    rest /= mean_val
    rest = rest.transpose(1, 0)
    rest = norm(rest)

    return rest


def image_stack(origin_image):
    # 灰度图像增强
    img1 = ft_set(flat(origin_image))

    # 形态学膨胀图
    kernel = np.ones((3, 3), np.uint8)
    img2 = cv2.dilate(img1, kernel, iterations=1)

    img0, img1, img2 = origin_image * 255, img1 * 255, img2 * 255

    # 堆叠
    img = np.stack((img0, img1, img2), 0)
    img = img.astype(np.float32)

    img /= 255

    img_new = np.stack((img0, img1, img2), 2)
    img_new /= 255

    # img_show = np.stack((img0, img1, img2), 2)
    # img_show = img_show.astype(np.float32)
    #
    # img_show /= 255
    #
    # draw_single(img_show)

    return img


def filter_rectangle_area(prop: RegionProperties, params: SDParams) -> bool:
    """
    按照面积比率过滤
    """
    fill_percent = prop.area_filled / prop.area_bbox
    # 区域面积比例大于阈值则可能为斜线
    if fill_percent > params.box_fill_threshold:
        return True
    return False


def filter_line_gradient(prop: RegionProperties) -> bool:
    """
    按照直线斜率过滤
    """
    y_a, x_a, y_b, x_b = prop.bbox
    width, height = math.fabs(x_a - x_b), math.fabs(y_a - y_b)

    return False


def filter_total_height_and_weight(
        slope_candidates: List[tuple[float, float, float, float]],
        width: float,
        height: float,
        params: SDParams
) -> bool:
    """
    按照横纵坐标总占比过滤
    """
    width_delta, height_delta = 0, 0
    for cand in slope_candidates:
        y_a, x_a, y_b, x_b = cand
        height_delta += math.fabs(y_a - y_b)
        width_delta += math.fabs(x_a - x_b)

    height_fill_percent = height_delta / height
    width_fill_percent = width_delta / width
    if height_fill_percent < params.box_projection_threshold and width_fill_percent < params.box_projection_threshold:
        return True

    return False


def analyze_shape(predict_img: np.ndarray, params: SDParams):
    """连通区域标记，消除孤立噪点"""
    # 8连通区域标记
    labels = measure.label(predict_img, connectivity=2)
    props = measure.regionprops(labels, intensity_image=None)
    height, width = predict_img.shape

    slope_candidates = []
    for prop in props:
        if filter_rectangle_area(prop, params):
            continue
        # if filter_line_gradient(prop):
        #     continue
        slope_candidates.append(prop.bbox)

    if filter_total_height_and_weight(slope_candidates, width, height, params):
        return []

    return slope_candidates


def extend_origin_image(
        image_pre,
        image,
        pre_window_start,
        window_start,
        pre_window_end,
        window_end
):
    height, width_pre = image_pre.shape
    _, width = image.shape

    width_per_second = int(float(width_pre) / float(pre_window_end - pre_window_start))

    new_start_time = int(min(pre_window_start, window_start))
    new_end_time = int(max(pre_window_end, window_end))

    new_width = width_per_second * (new_end_time - new_start_time)

    new_image = np.array(np.zeros((height, new_width)))

    offset_start = 0

    if pre_window_start < window_start:
        # 起始时间不同需要offset
        offset_start = int((window_start - pre_window_start) * width_per_second)

    # 拷贝A图
    y_a, x_a, y_b, x_b = 0, 0, height, width_pre
    new_image[y_a:y_b, x_a:x_b] = image_pre[y_a:y_b, x_a:x_b]

    # 拷贝B图
    y_a, x_a, y_b, x_b = 0, 0, height, width
    new_image[y_a:y_b, x_a + offset_start:x_b + offset_start] = image[y_a:y_b, x_a:x_b]

    return new_image


def concat_image(
        image_pre,
        image,
        position_pre,
        position,
        pre_window_start,
        window_start,
        pre_window_end,
        window_end
):
    height, width_pre = image_pre.shape
    height, width = image.shape

    offset_start = 0

    width_per_second = int(float(width_pre) / float(pre_window_end - pre_window_start))

    new_start_time = int(min(pre_window_start, window_start))
    new_end_time = int(max(pre_window_end, window_end))

    new_width = width_per_second * (new_end_time - new_start_time)

    new_image = np.array(np.zeros((height, new_width)))

    if pre_window_start < window_start:
        # 起始时间不同需要offset
        offset_start = int((window_start - pre_window_start) * width_per_second)

    # 拷贝A图
    pre_y_a, pre_x_a, pre_y_b, pre_x_b = position_pre
    new_image[pre_y_a:pre_y_b, pre_x_a:pre_x_b] = image_pre[pre_y_a:pre_y_b, pre_x_a:pre_x_b]

    # 拷贝B图
    y_a, x_a, y_b, x_b = position
    new_image[y_a:y_b, x_a + offset_start:x_b + offset_start] = image[y_a:y_b, x_a:x_b]

    return new_image

def _cmp_func(a, b):
    if a[4] != b[4]:
        # 先按照窗口时间排序
        return a[4] > b[4]
    else:
        # 再按照toa排序
        return a[3] > b[3]


def concat_slopes(detected_slopes):
    """
    合并同范围斜线轮廓
    """

    # 按照toa排序
    ordered_candidates = sorted(detected_slopes, key=functools.cmp_to_key(_cmp_func))

    pre_candidate = None

    i = 0
    while i < len(ordered_candidates):
        if i == 0:
            pre_candidate = ordered_candidates[i]
            i += 1
        else:
            candidate = ordered_candidates[i]
            pre_positions, pre_skeleton_image, pre_origin_image, pre_toa, pre_window_start, pre_window_end = pre_candidate
            positions, skeleton_image, origin_image, toa, window_start, window_end = candidate

            pre_y_a, pre_x_a, pre_y_b, pre_x_b = pre_positions
            y_a, x_a, y_b, x_b = positions

            _, width_pre = pre_skeleton_image.shape
            _, width = skeleton_image.shape

            width_per_second = int(float(width_pre) / float(pre_window_end - pre_window_start))

            offset_width = width_pre - width

            if pre_window_start != window_start:
                # 不在同一个时间分片
                offset_width = int((window_start - pre_window_start) * width_per_second)

            if pre_window_start <= window_start <= pre_window_end:
                # print("concat image with previous, current time: {} ~ {}, previous time: {} ~ {}".format(
                #     window_start,
                #     window_end,
                #     pre_window_start,
                #     pre_window_end
                # ))

                # 时间能接上则连接
                new_skeleton = concat_image(
                    pre_skeleton_image,
                    skeleton_image,
                    pre_positions,
                    positions,
                    pre_window_start,
                    window_start,
                    pre_window_end,
                    window_end,
                )

                new_origin = extend_origin_image(
                    pre_origin_image,
                    origin_image,
                    pre_window_start,
                    window_start,
                    pre_window_end,
                    window_end
                )

                # draw_triple(pre_skeleton_image, skeleton_image, new_skeleton, title_a=pre_window_start, title_b=window_start)

                new_window_end = int(max(pre_window_end, window_end))
                x_a += offset_width
                x_b += offset_width

                #  y_a, x_a, y_b, x_b
                concat_positions = (min(y_a, pre_y_a), min(x_a, pre_x_a), max(y_b, pre_y_b), max(x_b, pre_x_b))

                new_candidate = (
                    concat_positions,
                    new_skeleton, new_origin, pre_toa, pre_window_start, new_window_end)

                # 扩展上一个候选
                ordered_candidates[i - 1] = new_candidate

                # 删除当前候选
                ordered_candidates.remove(candidate)

                pre_candidate = new_candidate
            else:
                pre_candidate = candidate
                i += 1

    return ordered_candidates
