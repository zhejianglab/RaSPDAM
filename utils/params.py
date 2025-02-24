"""
时间窗口采样间隔和步长
"""

DEFAULT_OUTPUT_PATH = ""
DEFAULT_MODEL_PATH = "models"
DEFAULT_BOX_FILL_PERCENT_THRESHOLD = 0.4
DEFAULT_IOU_THRESHOLD = 0.5
DEFAULT_OVERLAP_THRESHOLD = 0.5
DEFAULT_PROJECTION_PERCENT_THRESHOLD = 0.05
DEFAULT_TIME_WINDOW_SIZE = 2


class SDParams:
    def __init__(self,
                 model_path=DEFAULT_MODEL_PATH,
                 output_path=DEFAULT_OUTPUT_PATH,
                 iou_threshold=DEFAULT_IOU_THRESHOLD,
                 overlap_threshold=DEFAULT_OVERLAP_THRESHOLD,
                 box_fill_threshold=DEFAULT_BOX_FILL_PERCENT_THRESHOLD,
                 box_projection_threshold=DEFAULT_PROJECTION_PERCENT_THRESHOLD,
                 time_window_size=DEFAULT_TIME_WINDOW_SIZE
                 ) -> None:
        self.model_path = model_path
        self.output_path = output_path
        self.iou_threshold = iou_threshold
        self.overlap_threshold = overlap_threshold
        self.box_fill_threshold = box_fill_threshold
        self.box_projection_threshold = box_projection_threshold
        self.time_window_size = time_window_size
