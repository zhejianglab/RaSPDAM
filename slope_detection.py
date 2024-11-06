import math
import os
import time

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn import DataParallel

from analyzer import calc_dm
from fitsreader import FitsReader
from image_transform import analyze_shape, image_resize, image_stack, draw_single, draw_double
import argparse

from params import SDParams, DEFAULT_SIGMOID_THRESHOLD, DEFAULT_OUTPUT_PATH, DEFAULT_MODEL_PATH, \
    DEFAULT_BOX_FILL_PERCENT_THRESHOLD, DEFAULT_PROJECTION_PERCENT_THRESHOLD, TIME_WINDOW_STEP, TIME_WINDOW_SIZE


class TimeSeriesSlice:
    def __init__(self, raw_image: np.ndarray, stack_image: np.ndarray, start_time: float, end_time: float) -> None:
        self.raw_image = raw_image
        self.stack_image = stack_image
        self.start_time = start_time
        self.end_time = end_time


def draw_result_image(output_path: str,
                      file_name: str,
                      origin_image: np.ndarray,
                      slope_skeleton: np.ndarray,
                      dm_calc_result: dict,
                      fits_data: FitsReader
                      ):
    # 画白框
    origin_image_with_box = origin_image.copy()

    y_a, x_a, y_b, x_b = dm_calc_result["box"]
    origin_image_with_box[y_a:y_b, x_a - 2:x_a] = 1
    origin_image_with_box[y_a:y_b, x_b:x_b + 2] = 1
    origin_image_with_box[y_a - 2:y_a, x_a:x_b] = 1
    origin_image_with_box[y_b:y_b + 2, x_a:x_b] = 1

    # ======= 结果图拼接开始 ========

    height, width = origin_image_with_box.shape

    toa = dm_calc_result["t1"]
    dm = dm_calc_result["dm"]
    window_start = dm_calc_result["t_start"]
    window_end = dm_calc_result["t_end"]
    freq_start = dm_calc_result["freq1"]
    freq_end = dm_calc_result["freq2"]

    result_file_name = "{}_{}_{}.png".format(file_name, round(toa, 2), round(dm, 2))
    result_file_path = os.path.join(output_path, result_file_name)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), dpi=130)

    plt.suptitle(file_name)

    ax1.imshow(origin_image_with_box)
    ax1.set_xticks(np.linspace(0, width, 4), np.around(np.linspace(window_start, window_end, 4), 2))
    ax1.set_xlabel('time(s)')
    ax1.set_yticks(np.linspace(0, height, 4), np.around(np.linspace(freq_start, freq_end, 4), 2))
    ax1.set_ylabel('Frequency(MHz)')
    ax1.set_title("Origin")

    ax2.imshow(slope_skeleton)
    ax2.set_xticks(np.linspace(0, width, 4), np.around(np.linspace(window_start, window_end, 4), 2))
    ax2.set_yticks(np.linspace(0, height, 4), np.around(np.linspace(freq_start, freq_end, 4), 2))
    ax2.set_title("Slope Skeleton")

    text_verbose = "t1 = {} s , t2 = {} s , freq1 = {} MHz , freq2 = {} MHz".format(
        round(dm_calc_result["t1"], 2),
        round(dm_calc_result["t2"], 2),
        round(dm_calc_result["freq1"], 2),
        round(dm_calc_result["freq2"], 2)
    )
    fig.text(0.5, 0.03, text_verbose, fontsize=11, ha='center', va='center')

    fig.text(0.5, 0.07, "DM = {} $pc\,cm^-3$ , toa = {} s".format(round(dm, 2), round(toa, 2)),
             fontsize=12,
             ha='center', va='center')

    title_info = "time sampling: {} us, freq sampling: {} MHz".format(
        round(fits_data.tsamp * math.pow(10, 6), 2),
        fits_data.freq)

    fig.text(0.5, 0.93, "time window: {} s - {} s".format(window_start, window_end), fontsize=10, ha='center',
             va='center')

    fig.text(0.5, 0.90, title_info, fontsize=10, ha='center', va='center')

    fig.text(0.5, 0.87, "RA: {}, DEC: {}".format(fits_data.ra, fits_data.dec), fontsize=10, ha='center', va='center')

    plt.savefig(result_file_path)


class SlopeDetection:

    def __init__(self, params: SDParams) -> None:

        self.params = params

        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')

        # 加载模型
        # self.model = CE_Net_res34_PCAM_009("models")

        model_path = os.path.join(params.model_path, "net.pth")

        print("Start load model, path: {}".format(model_path))

        # self.model = torch.load(model_path)
        self.model = torch.load(model_path, device)
        # torch.save(self.model,"net.pth")

        print("Model has been loaded")

        self.model = DataParallel(self.model)

    def segment(self, fit_file):
        fit_file_name = fit_file[fit_file.rfind("/") + 1:fit_file.rfind(".fits")]

        with FitsReader(fit_file).readFAST() as fits:
            # 单个Fits文件读取

            freq_list = fits.chan_freqs
            sub_data = fits.getdata()

            x_max, y_max = sub_data.shape

            total_time = fits.nline * fits.nsblk * fits.tsamp
            resolution_per_second = int(round(x_max / total_time))

            start_time = time.time()
            time_slice_list = []

            for i in range(0, int(math.floor(total_time)), TIME_WINDOW_STEP):
                x_start, x_end = i * resolution_per_second, (i + TIME_WINDOW_SIZE) * resolution_per_second

                if x_end > x_max:
                    x_end = x_max
                image_data = sub_data[x_start:x_end]

                # time1 = time.time()
                # 图像大小压缩成512*512，并卷积堆叠
                raw_image = image_resize(image_data)

                # time2 = time.time()

                stack_image = image_stack(raw_image)
                # time3 = time.time()

                # print("image_resize: {}s, image_stack: {}s".format(time2 - time1, time3 - time2))

                time_slice_list.append(TimeSeriesSlice(
                    raw_image,
                    stack_image,
                    x_start / resolution_per_second,
                    x_end / resolution_per_second))

            end_time = time.time()

            print("Image transformation finish, cost: {}s".format(round(end_time - start_time, 2)))

            tensor_input = torch.Tensor(np.array([e.stack_image for e in time_slice_list]))

            model_output = self.model(tensor_input).squeeze(1).cpu()

            model_result = torch.sigmoid(model_output) > self.params.sigmoid_threshold
            # model_result = model_output.detach().numpy()
            count, img_h, img_w = model_result.shape

            for i in range(count):
                target_slice = time_slice_list[i]

                model_shape_image = model_result[i]

                # draw_double(target_slice.raw_image, model_shape_image)

                # 后处理以及过滤，标记所有目标框
                slope_candidates = analyze_shape(model_shape_image, self.params)

                if len(slope_candidates) == 0:
                    # 不存在则跳过
                    continue

                pure_slope_image = np.array(np.zeros((img_h, img_w)))
                for position in slope_candidates:
                    # 只保留检测区域，其他区域涂黑
                    y_a, x_a, y_b, x_b = position
                    pure_slope_image[y_a:y_b, x_a:x_b] = model_shape_image[y_a:y_b, x_a:x_b]

                calc_result = calc_dm(pure_slope_image, target_slice.start_time, target_slice.end_time, freq_list)

                # 绘制结果图
                draw_result_image(
                    self.params.output_path,
                    fit_file_name,
                    target_slice.raw_image,
                    pure_slope_image,
                    calc_result,
                    fits
                )

        # print("========================================================\n")

    def detect(self, fit_path):
        start_time = time.time()

        self.model.eval()

        if not os.path.exists(fit_path):
            print("path: {} is not a valid fits file or directory".format(fit_path))
            exit(-1)

        if os.path.isdir(fit_path):
            print("running in walkdir mode")
            # 使用进程池批量运行

            fit_files = []

            for root, dirs, files in os.walk(fit_path):
                for name in files:
                    if name.endswith("fits"):
                        fit_files.append(os.path.join(root, name))

                for i in range(len(fit_files)):
                    print("<{}/{}>Handling fits file: {}".format(i + 1, len(fit_files), fit_files[i]))

                    self.segment(fit_files[i])

        else:
            print("running in single file mode")

            # 单个运行
            self.segment(fit_path)

        end_time = time.time()
        time_cost = round(end_time - start_time, 2)

        print("\n--------------------------------------------------------")
        print("TIME_COST: {}s".format(time_cost))
        print("--------------------------------------------------------\n")


if __name__ == '__main__':
    print("==================== Environment Check ====================")
    print("OpenCV CUDA Devices: {}".format(cv2.cuda.getCudaEnabledDeviceCount()))
    print("PyTorch CUDA Available: {}".format(torch.cuda.is_available()))
    print("===========================================================")

    parser = argparse.ArgumentParser()

    parser.add_argument("fits_file")

    parser.add_argument(
        "-m", "-model_path", type=str,
        default=DEFAULT_MODEL_PATH
    )
    parser.add_argument(
        "-o", "-output_path", type=str,
        default=DEFAULT_OUTPUT_PATH
    )

    parser.add_argument(
        "-sigmoid_threshold", type=float,
        default=DEFAULT_SIGMOID_THRESHOLD
    )

    parser.add_argument(
        "-box_fill_threshold", type=float,
        default=DEFAULT_BOX_FILL_PERCENT_THRESHOLD
    )

    parser.add_argument(
        "-box_projection_threshold", type=float,
        default=DEFAULT_PROJECTION_PERCENT_THRESHOLD
    )

    opt = parser.parse_args()

    params = SDParams(
        opt.m,
        opt.o,
        opt.sigmoid_threshold,
        opt.box_fill_threshold,
        opt.box_projection_threshold
    )

    t = SlopeDetection(params)

    t.detect(opt.fits_file)
