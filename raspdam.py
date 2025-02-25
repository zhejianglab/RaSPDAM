import argparse
import math
import os
import queue
import threading
import time

import numpy as np

from utils.analyzer import calc_dm, non_max_overlap_suppression
from utils.fitsreader import FitsReader
from utils.filreader import FilReader
from utils.image_transform import analyze_shape, image_resize, image_stack

from utils.params import SDParams, DEFAULT_OUTPUT_PATH, DEFAULT_MODEL_PATH, \
    DEFAULT_IOU_THRESHOLD, DEFAULT_OVERLAP_THRESHOLD, DEFAULT_BOX_FILL_PERCENT_THRESHOLD, \
    DEFAULT_TIME_WINDOW_SIZE, DEFAULT_PROJECTION_PERCENT_THRESHOLD

import matplotlib.pyplot as plt
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from tqdm import tqdm

# nnUNet_results = 'models/nnUNet_trained_models'

ENABLE_CUPY = False


class TimeSeriesSlice:
    def __init__(self, raw_image: np.ndarray, stack_image: np.ndarray, start_time: float, end_time: float, nslice: int, total_slice:int) -> None:
        self.raw_image = raw_image
        self.stack_image = stack_image
        self.start_time = start_time
        self.end_time = end_time
        self.nslice = nslice
        self.total_slice = total_slice


def draw_result_image(output_path, file_name, origin_image, enhanced_image, dm_calc_result, obs):
    # 画白框
    origin_image_with_box = origin_image.copy() * 255
    enhanced_image_with_box = enhanced_image.copy()

    y_a, x_a, y_b, x_b = dm_calc_result["box"]
    origin_image_with_box[y_a:y_b, x_a - 2:x_a] = 255
    origin_image_with_box[y_a:y_b, x_b:x_b + 2] = 255
    origin_image_with_box[y_a - 2:y_a, x_a:x_b] = 255
    origin_image_with_box[y_b:y_b + 2, x_a:x_b] = 255

    enhanced_image_with_box[y_a:y_b, x_a - 2:x_a] = 255
    enhanced_image_with_box[y_a:y_b, x_b:x_b + 2] = 255
    enhanced_image_with_box[y_a - 2:y_a, x_a:x_b] = 255
    enhanced_image_with_box[y_b:y_b + 2, x_a:x_b] = 255

    # ======= 结果图拼接开始 ========

    height, width = origin_image_with_box.shape

    toa = dm_calc_result["toa"]
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

    ax2.imshow(enhanced_image_with_box)
    ax2.set_xticks(np.linspace(0, width, 4), np.around(np.linspace(window_start, window_end, 4), 2))
    ax2.set_yticks(np.linspace(0, height, 4), np.around(np.linspace(freq_start, freq_end, 4), 2))
    ax2.set_title("Enhanced")

    text_verbose = "t1 = {} s , t2 = {} s , freq1 = {} MHz , freq2 = {} MHz".format(
        round(dm_calc_result["t1"], 2),
        round(dm_calc_result["t2"], 2),
        round(dm_calc_result["freq1"], 2),
        round(dm_calc_result["freq2"], 2)
    )
    fig.text(0.5, 0.03, text_verbose, fontsize=11, ha='center', va='center')

    fig.text(0.5, 0.07, "DM = {} $pc,cm^-3$ , toa = {} s".format(round(dm, 2), round(toa, 2)),
             fontsize=12,
             ha='center', va='center')

    title_info = "time sampling: {} us, freq sampling: {} MHz".format(
        round(obs.tsamp * math.pow(10, 6), 2),
        obs.freq)

    fig.text(0.5, 0.93, "time window: {} s - {} s".format(window_start, window_end), fontsize=10, ha='center',
             va='center')

    fig.text(0.5, 0.90, title_info, fontsize=10, ha='center', va='center')

    fig.text(0.5, 0.87, "RA: {}, DEC: {}".format(obs.ra, obs.dec), fontsize=10, ha='center', va='center')

    plt.savefig(result_file_path)
    plt.close()


def drawing_handle(q):
    while True:
        try:
            item = q.get()
            if item is None:  # 明确的终止条件
                return

            try:
                output_path, fit_file_name, raw_image, enhanced_image, calc_result, obs = item
                draw_result_image(
                    output_path,
                    fit_file_name,
                    raw_image,
                    enhanced_image,
                    calc_result,
                    obs
                )
            except Exception as e:
                print(f"Error processing item {item}: {e}")
                raise  # 重新抛出异常以便外层处理
        except Exception as e:
            print(f"Error in drawing_handle: {e}")
        finally:
            q.task_done()

def normalize_box(calc_result):
    x_min = calc_result["t1"]
    x_max = calc_result["t2"]
    y_min = calc_result["freq2"]
    y_max = calc_result["freq1"]
    # 确保 x_min <= x_max 和 y_min <= y_max
    x_min, x_max = min(x_min, x_max), max(x_min, x_max)
    y_min, y_max = min(y_min, y_max), max(y_min, y_max)
    return [x_min, y_min, x_max, y_max]


def segment_pool_handle(predictor, seg_q, drawing_q, candidate_pool, lock, params):
    while True:
        try:
            item = seg_q.get()
            if item is None:
                return

            file_name, freq_list, obs, time_slice, pbar, time_window_step = item
            calc_result = handle_candidate(predictor, params, freq_list, time_slice, pbar)
            pbar.update(time_window_step)

            with lock:
                if file_name not in candidate_pool:
                    candidate_pool[file_name] = []
                candidate_pool[file_name].append((calc_result, time_slice))

            if len(candidate_pool[file_name]) < time_slice.total_slice:
                continue

            print("candidate_pool: {}, total_slice: {}".format(len(candidate_pool[file_name]), time_slice.total_slice))

            calc_results = [(calc_result, time_slice) for calc_result, time_slice in candidate_pool[file_name] if calc_result]

            print("calc_results: {}".format(calc_results))

            if not calc_results:
                continue

            boxes = np.array([normalize_box(calc_result) for calc_result, time_slice in calc_results], dtype=np.float32)
            scores = np.array([calc_result["score"] for calc_result, time_slice in calc_results], dtype=np.float32)

            if len(boxes) == 0 or len(scores) == 0:
                continue

            print("boxes: {}, scores: {}".format(boxes, scores))
            indices = non_max_overlap_suppression(boxes, scores, params.iou_threshold, params.overlap_threshold)
            print("indices: {}".format(indices))
            filtered_calc_results = [calc_results[i] for i in indices]

            for calc_result, time_slice in filtered_calc_results:
                try:
                    drawing_q.put((
                        params.output_path,
                        file_name,
                        time_slice.raw_image,
                        time_slice.stack_image[2][0],
                        calc_result,
                        obs
                    ))
                except Exception as e:
                    print(f"Error putting item into drawing_q: {e}")
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Unexpected error in segment_pool_handle: {e}")
        finally:
            seg_q.task_done()


def handle_candidate(predictor, params, freq_list, time_slice, pbar):
    # with open("output_model.txt", 'w+') as f:
    #     print(predictor.network.summary())

    # summary(predictor.network, (1, 512, 512))
    # torch.onnx.export(predictor.network, time_slice.stack_image.tolist(), "unet.onnx", verbose=True)

    nnunet_ret = None
    with torch.inference_mode():
        nnunet_ret = predictor.predict_single_npy_array(
            time_slice.stack_image,
            {'spacing': (999, 1, 1)}
        )

    model_shape_image = nnunet_ret[0]

    # draw_single_file(model_shape_image, time_slice.start_time, time_slice.end_time)

    img_h, img_w = model_shape_image.shape

    slope_candidates = analyze_shape(model_shape_image, params)

    if len(slope_candidates) == 0:
        # 不存在则跳过s
        return None

    pure_slope_image = np.array(np.zeros((img_h, img_w)))
    for position in slope_candidates:
        # 只保留检测区域，其他区域涂黑
        y_a, x_a, y_b, x_b = position
        pure_slope_image[y_a:y_b, x_a:x_b] = model_shape_image[y_a:y_b, x_a:x_b]

    calc_result = calc_dm(pure_slope_image, time_slice.start_time, time_slice.end_time, freq_list, pbar)

    return calc_result
    # 绘制结果图
    # draw_result_image(
    #     params.output_path,
    #     fit_file_name,
    #     time_slice.raw_image,
    #     pure_slope_image,
    #     calc_result,
    #     fits_reader
    # )


class RaSPDAM:

    def __init__(self, params: SDParams) -> None:
        self.params = params
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("Pytorch and Numpy will be initialized CUDA mode")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Pytorch and Numpy will be initialized MPS mode")

        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False
        )

        predictor.initialize_from_trained_model_folder(
            params.model_path,
            use_folds=None,
            checkpoint_name="checkpoint_final.pth",
        )

        self.device = device
        self.predictor = predictor
        self.candidate_pool = {}
        self.lock = threading.Lock()

    def segment(self, file):
        file_name, file_extension = os.path.splitext(os.path.basename(file))
        file_extension = file_extension[1:]
        if file_extension == "fil":
            reader = FilReader
        elif file_extension == "fits":
            reader = FitsReader
        else:
            raise Exception("Unsupported file extension: {}".format(file_extension))

        segment_q = queue.Queue(maxsize=10)
        drawing_q = queue.Queue(maxsize=10)

        background_drawing_thread = threading.Thread(target=drawing_handle,
                                                     args=[drawing_q],
                                                     daemon=False)
        background_drawing_thread.start()

        background_thread = threading.Thread(target=segment_pool_handle,
                                             args=[self.predictor, segment_q, drawing_q, self.candidate_pool, self.lock, self.params],
                                             daemon=False)
        background_thread.start()

        with reader(file) as obs:
            freq_list = obs.chan_freqs

            total_time_seconds = obs.total_time_seconds

            time_window_size = self.params.time_window_size
            # 每次迭代，都有一半窗口重叠
            time_window_step = int(math.floor(time_window_size / 2))

            sliding_window_end = int(math.ceil(total_time_seconds))
            total_slices = int(math.ceil(total_time_seconds / time_window_step))

            pbar = tqdm(total=sliding_window_end, desc=file_name)
            for i in range(0, sliding_window_end, time_window_step):
                # 按窗口读取部分数据
                image_data = obs.read_data(i, i + time_window_size)

                # time_start = time.time()
                # 图像大小压缩成512*512，并卷积堆叠
                raw_image = image_resize(image_data)
                stack_image = image_stack(raw_image)
                # time_end = time.time()

                # draw_list_file(
                #     [stack_image[0][0], stack_image[1][0], stack_image[2][0]],
                #     int(x_start / resolution_per_second)
                # )

                # print("image transformation: {} ms".format(round(time_end - time_start, 2) * 1000))

                start_time, end_time = i, i + time_window_size
                if end_time > total_time_seconds:
                    end_time = total_time_seconds
                time_slice = TimeSeriesSlice(
                    raw_image,
                    stack_image,
                    start_time,
                    end_time,
                    i,
                    total_slices
                )
                segment_q.put((file_name, freq_list, obs, time_slice, pbar, time_window_step))

        segment_q.join()
        segment_q.put(None)
        drawing_q.join()
        drawing_q.put(None)
        background_thread.join()
        background_drawing_thread.join()

        pbar.close()

    def detect(self, path):
        start_time = time.time()

        if not os.path.exists(path):
            print("path: {} is not a valid file or directory".format(path))
            exit(-1)

        if os.path.isdir(path):
            print("running in walkdir mode")
            obs_files = []

            for root, dirs, files in os.walk(path):
                for name in files:
                    if name.endswith("fits") or name.endswith("fil"):
                        obs_files.append(os.path.join(root, name))

                for i in range(len(obs_files)):
                    print("<{}/{}>Handling observation file: {}".format(i + 1, len(obs_files), obs_files[i]))

                    self.segment(obs_files[i])

        else:
            print("running in single file mode")
            self.segment(path)

        end_time = time.time()
        time_cost = round(end_time - start_time, 2)

        print("\n--------------------------------------------------------")
        print("TIME_COST: {}s".format(time_cost))
        print("--------------------------------------------------------\n")


if __name__ == '__main__':
    print("==================== Environment Check ====================")
    print("PyTorch CUDA Available: {}".format(torch.cuda.is_available()))
    print("===========================================================")

    parser = argparse.ArgumentParser()

    parser.add_argument("path")

    parser.add_argument(
        "-m", "-model_path", type=str,
        default=DEFAULT_MODEL_PATH
    )
    parser.add_argument(
        "-o", "-output_path", type=str,
        default=DEFAULT_OUTPUT_PATH
    )

    parser.add_argument(
        "-box_fill_threshold", type=float,
        default=DEFAULT_BOX_FILL_PERCENT_THRESHOLD
    )

    parser.add_argument(
        "-iou_threshold", type=float,
        default=DEFAULT_IOU_THRESHOLD
    )

    parser.add_argument(
        "-overlap_threshold", type=float,
        default=DEFAULT_OVERLAP_THRESHOLD
    )

    parser.add_argument(
        "-box_projection_threshold", type=float,
        default=DEFAULT_PROJECTION_PERCENT_THRESHOLD
    )

    parser.add_argument(
        "-time_window_size", type=int,
        default=DEFAULT_TIME_WINDOW_SIZE
    )

    opt = parser.parse_args()

    params = SDParams(
        opt.m,
        opt.o,
        opt.iou_threshold,
        opt.overlap_threshold,
        opt.box_fill_threshold,
        opt.box_projection_threshold,
        opt.time_window_size
    )

    t = RaSPDAM(params)

    t.detect(opt.path)
