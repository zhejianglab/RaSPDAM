import math

import numpy as np
from skimage import measure


def calc_dm(pure_slope_image: np.ndarray, start_time: float, end_time: float, freq_list: list) -> dict:
    # 再次连通区域分析，用面积最大的区域做计算
    labels = measure.label(pure_slope_image, connectivity=2)
    props = measure.regionprops(labels, intensity_image=None)

    result = None
    max_area = -999

    # 取面积最大区域计算
    for prop in props:
        y_a, x_a, y_b, x_b = prop.bbox
        # 计算面积
        area = int(math.fabs(y_a - y_b) * math.fabs(x_a - x_b))
        if area > max_area:
            max_area = area
            result = prop.bbox

    y_a, x_a, y_b, x_b = result

    height, width = pure_slope_image.shape

    widow_delta_t = end_time - start_time

    freq_delta = math.floor(len(freq_list) / height)

    t1 = start_time + (widow_delta_t * x_a / width)
    t2 = start_time + (widow_delta_t * x_b / width)

    delta_t = math.fabs(t2 - t1)

    freq_index_1 = freq_delta * (height - y_a)
    freq_index_2 = freq_delta * (height - y_b)
    freq1 = float(freq_list[len(freq_list) - 1 if freq_index_1 >= len(freq_list) else freq_index_1])
    freq2 = float(freq_list[len(freq_list) - 1 if freq_index_2 >= len(freq_list) else freq_index_2])

    dm = 2.41 * math.pow(10, -4) * delta_t * math.pow((1 / math.pow(freq2, 2)) - (1 / math.pow(freq1, 2)), -1)

    print("t1: {}, t2: {}, freq1: {}, freq2: {}, dm: {}".format(t1, t2, freq1, freq2, dm))

    return {
        "box": (y_a, x_a, y_b, x_b),
        "dm": dm,
        "t1": t1,
        "t2": t2,
        "freq1": freq1,
        "freq2": freq2,
        "t_start": start_time,
        "t_end": end_time,
    }
