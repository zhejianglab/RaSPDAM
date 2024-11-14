from re import L
import numpy as np
import cupy as cp
import cv2

# 通用卷积核斜率
secant = [
    1.3125,
    1.09375,
    0.9375,
    0.78125,
    0.671875,
    0.59375,
    0.515625,
    0.4603174603174603,
]


def norm_cu(x):
    denominator = cp.max(x) - cp.min(x)
    if denominator == 0:
        return (x - cp.min(x)) / 1
    return (x - cp.min(x)) / denominator


def flat_cu(res):
    flat_data = cp.sort(res.flatten())
    tshd = 1
    vmin, vmax = (
        flat_data[50 * len(flat_data) // 100],
        flat_data[(100 - tshd) * len(flat_data) // 100],
    )
    res = cp.clip(res, vmin, vmax)
    res = norm_cu(res)
    return res


def ft_set_cu(res):
    rest = cp.zeros_like(res)
    cut = 30
    k_set = secant[::-1]
    x_p = cp.floor(cp.linspace(0, cut - 1, cut)).astype(cp.int32)  # 生成右半部分自变量
    x_n = -x_p[::-1]  # 生成左半部分自变量

    split = 0
    tem = 512 // len(k_set)

    for k in k_set:  # 生成不同斜率的卷积核
        km = k if k <= 1.0 else 1 / k

        y_p = cp.floor(km * x_p).astype(cp.int32)  # 根据 x坐标、斜率k --> y坐标
        y_n = -y_p[::-1]

        for ker_size in range(3, 18, 2):  # 生成不同尺寸卷积核
            ker = cp.zeros((ker_size, ker_size))
            num = ker_size // 2

            x_n_tem, x_p_tem = x_n[-num - 1 :] + num, x_p[: num + 1] + num
            y_n_tem, y_p_tem = y_n[-num - 1 :] + num, y_p[: num + 1] + num

            ker[x_n_tem, y_n_tem] = 1.0  # 赋值卷积核左半部分直线
            ker[x_p_tem, y_p_tem] = 1.0  # 赋值卷积核右半部分直线

            f0_num = cp.sum(ker == 0)
            f1_num = cp.sum(ker == 1)
            ker[ker == 0] = -f1_num / f0_num  # 赋值其他区域 使得卷积核总和为0

            if k > 1.0:
                ker = ker.transpose(1, 0)

            # 切片 卷积 叠加
            rest[split * tem : (split + 1) * tem] += cp.abs(
                cp.asarray(
                    cv2.filter2D(
                        cp.asnumpy(res)[split * tem : (split + 1) * tem],
                        -1,
                        kernel=cp.asnumpy(ker),
                    )
                )
            )

        split += 1

    # 后处理
    rest = rest.transpose(1, 0)
    mean_val = cp.mean(rest, axis=0)
    mean_val[mean_val == 0.0] = 1.0
    rest /= mean_val
    rest = rest.transpose(1, 0)
    rest = norm_cu(rest)

    return rest


def image_stack_cu(origin_image_cu):
    # 灰度图像增强
    img1 = ft_set_cu(flat_cu(origin_image_cu))

    # 形态学膨胀图
    img2 = cp.asarray(
        cv2.dilate(cp.asnumpy(img1), np.ones((3, 3), np.uint8), iterations=1)
    )

    img0, img1, img2 = origin_image_cu * 255, img1 * 255, img2 * 255

    # 堆叠
    img = cp.stack((img0, img1, img2), 0)
    img = img.astype(cp.float32)

    img /= 255

    return img


def resize_norm_cu(x):
    return (x - cp.min(x)) / (cp.max(x) - cp.min(x))


def image_process_cu(input_image):
    input_image_cu = cp.asarray(input_image)
    raw_image = image_resize_cu(input_image_cu)
    stack_image = image_stack_cu(raw_image)
    return cp.asnumpy(raw_image), cp.asnumpy(stack_image)


def image_resize_cu(input_image):
    image_data = input_image.copy()

    cut = 512  # resize的大小
    h, w = image_data.shape

    ch, cw = h // cut, w // cut
    # 多个像素点取均值，转为512*512
    image_data = cp.resize(image_data, (cut, ch, cut, cw))
    # image_data.resize(cut, ch, cut, cw)
    image_rescaled = cp.mean(image_data, axis=(1, 3))
    image_rescaled = resize_norm_cu(image_rescaled)
    image_rescaled -= cp.mean(image_rescaled, axis=0)
    image_rescaled = resize_norm_cu(image_rescaled)
    res = cp.rot90(image_rescaled)
    flat_data = cp.sort(image_rescaled.flatten())
    tshd = 5
    vmin, vmax = (
        flat_data[20 * len(flat_data) // 100],
        flat_data[(100 - tshd) * len(flat_data) // 100],
    )
    res = cp.clip(res, vmin, vmax)
    res = resize_norm_cu(res)

    return res
