import math
import numpy as np

from skimage import measure


def calc_dm(pure_slope_image, start_time, end_time, freq_list, pbar):
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

    dm = 2.40963855 * math.pow(10, -4) * delta_t * math.pow((1 / math.pow(freq2, 2)) - (1 / math.pow(freq1, 2)), -1)
    toa1 = t1 + 4.15 * 1000 * dm * (1 / math.pow(freq_list[len(freq_list) - 1], 2) - 1 / math.pow(freq1, 2))
    toa2 = t2 + 4.15 * 1000 * dm * (1 / math.pow(freq_list[len(freq_list) - 1], 2) - 1 / math.pow(freq2, 2))
    toa = (toa1 + toa2) / 2
    print("dm: {}, toa1: {}, toa2: {}, toa: {}".format(dm, toa1, toa2, toa))
    pbar.write("\nt1: {}, t2: {}, freq1: {}, freq2: {}, dm: {}".format(t1, t2, freq1, freq2, dm))

    return {
        "box": (y_a, x_a, y_b, x_b),
        "dm": dm,
        "t1": t1,
        "t2": t2,
        "freq1": freq1,
        "freq2": freq2,
        "t_start": start_time,
        "t_end": end_time,
        "toa": toa,
        "score": math.fabs(t2 - t1) * math.fabs(freq2 - freq1)
    }

def non_max_overlap_suppression(boxes, scores, threshold = 0.5, overlap_threshold = 0.5):
    """
    非极大值抑制
    :param boxes: 候选区域列表，每个区域由 (y_min, x_min, y_max, x_max) 表示
    :param scores: 每个候选区域的置信度分数
    :param threshold: NMS 阈值
    :param overlap_threshold: 交集占比阈值，超过该阈值则抑制分数较低的区域
    :return: 保留的区域索引
    """
    if len(boxes) == 0:
        return []

    # 按置信度分数排序
    sorted_indices = np.argsort(scores)[::-1]
    keep = []

    while len(sorted_indices) > 0:
        # 保留置信度最高的区域
        i = sorted_indices[0]
        keep.append(i)

        # 计算交并比（IoU）和交集占比
        ious = []
        overlaps = []
        for j in sorted_indices[1:]:
            iou = compute_iou(boxes[i], boxes[j])
            ious.append(iou)
            
            max_overlap = compute_overlap(boxes[i], boxes[j])
            overlaps.append(max_overlap)

        # 过滤掉 IoU 大于阈值的区域 或者 交集占比大于阈值的区域
        filtered_indices = np.where((np.array(ious) <= threshold) & (np.array(overlaps) <= overlap_threshold))[0]
        sorted_indices = sorted_indices[filtered_indices + 1]

    return keep

def compute_overlap(box1, box2):
    # 计算交集占比
    y_min1, x_min1, y_max1, x_max1 = box1
    y_min2, x_min2, y_max2, x_max2 = box2

    inter_y_min = max(y_min1, y_min2)
    inter_x_min = max(x_min1, x_min2)
    inter_y_max = min(y_max1, y_max2)
    inter_x_max = min(x_max1, x_max2)

    inter_area = max(0, inter_y_max - inter_y_min) * max(0, inter_x_max - inter_x_min)
    box1_area = (y_max1 - y_min1) * (x_max1 - x_min1)
    box2_area = (y_max2 - y_min2) * (x_max2 - x_min2)

    overlap1 = inter_area / box1_area
    overlap2 = inter_area / box2_area
    # 返回两个区域中交集占比的最大值，确保只要有一个区域的交集占比超过阈值就认为重叠
    return max(overlap1, overlap2)

def compute_iou(box1, box2):
    """
    计算两个区域的交并比（IoU）
    :param box1: 区域1，由 (y_min, x_min, y_max, x_max) 表示
    :param box2: 区域2，由 (y_min, x_min, y_max, x_max) 表示
    :return: IoU 值
    """
    y_min1, x_min1, y_max1, x_max1 = box1
    y_min2, x_min2, y_max2, x_max2 = box2

    # 计算交集区域
    inter_y_min = max(y_min1, y_min2)
    inter_x_min = max(x_min1, x_min2)
    inter_y_max = min(y_max1, y_max2)
    inter_x_max = min(x_max1, x_max2)

    inter_area = max(0, inter_y_max - inter_y_min) * max(0, inter_x_max - inter_x_min)

    # 计算并集区域
    box1_area = (y_max1 - y_min1) * (x_max1 - x_min1)
    box2_area = (y_max2 - y_min2) * (x_max2 - x_min2)
    union_area = box1_area + box2_area - inter_area

    # 计算 IoU
    iou = inter_area / union_area
    return iou
