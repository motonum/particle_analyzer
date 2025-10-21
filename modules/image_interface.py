import os
from modules.config import Config


class ImageInterface:
    def __init__(self, config: Config, image_path: str, parent_dir: str | None = None):
        self.config = config
        self.img_path = image_path
        self.filename = os.path.splitext(os.path.basename(image_path))[0]
        self.parent_dir = f"{parent_dir}/" if parent_dir else ""

    @property
    def output_detected_dir(self):
        return f"{self.config.OUTPUT_DIR_DETECTED}/{self.parent_dir}"

    @property
    def output_segmented_dir(self):
        return f"{self.config.OUTPUT_DIR_SEGMENTED}/{self.parent_dir}"

    @property
    def output_histogram_dir(self):
        return f"{self.config.OUTPUT_DIR_HISTOGRAM}/{self.parent_dir}"

    @property
    def output_csv_dir(self):
        return f"{self.config.OUTPUT_DIR_CSV}/{self.parent_dir}"

    @property
    def output_summary_dir(self):
        return f"{self.config.OUTPUT_DIR_SUMMARY}/{self.parent_dir}"

    @property
    def output_detected_path(self):
        return f"{self.output_detected_dir}/{self.filename}.tif"

    @property
    def output_segmented_path(self):
        return f"{self.output_segmented_dir}/{self.filename}.tif"

    def output_histogram_path(self, z: int | None):
        return f"{self.output_histogram_dir}{self.filename}-{'full' if z is None else f'z{z}'}.png"

    def output_csv_path(self, z: int | None):
        return f"{self.output_csv_dir}{self.filename}-{'full' if z is None else f'z{z}'}.csv"

    def output_summary_path(self, z: int | None):
        return f"{self.output_summary_dir}{self.filename}-{'full' if z is None else f'z{z}'}.csv"
