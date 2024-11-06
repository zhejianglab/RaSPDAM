"""
时间窗口采样间隔和步长
"""
TIME_WINDOW_SIZE = 2
TIME_WINDOW_STEP = 1

DEFAULT_OUTPUT_PATH = ""
DEFAULT_SIGMOID_THRESHOLD = 0.5
DEFAULT_MODEL_PATH = "models"
DEFAULT_BOX_FILL_PERCENT_THRESHOLD = 0.25
DEFAULT_PROJECTION_PERCENT_THRESHOLD = 0.15


class SDParams:
    def __init__(self,
                 model_path=DEFAULT_MODEL_PATH,
                 output_path=DEFAULT_OUTPUT_PATH,
                 sigmoid_threshold=DEFAULT_SIGMOID_THRESHOLD,
                 box_fill_threshold=DEFAULT_BOX_FILL_PERCENT_THRESHOLD,
                 box_projection_threshold=DEFAULT_PROJECTION_PERCENT_THRESHOLD
                 ) -> None:
        self.model_path = model_path
        self.output_path = output_path
        self.sigmoid_threshold = sigmoid_threshold
        self.box_fill_threshold = box_fill_threshold
        self.box_projection_threshold = box_projection_threshold
