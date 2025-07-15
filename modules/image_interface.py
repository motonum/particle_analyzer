import os


class ImageInterface:
    OUTPUT_DIR_DETECTED = "dist/detected"
    OUTPUT_DIR_SEGMENTED = "dist/segmented"
    OUTPUT_DIR_HISTOGRAM = "dist/histogram"

    def __init__(self, image_path: str):
        self.img_path = image_path
        self.filename = os.path.splitext(os.path.basename(image_path))[0]
    
    @property
    def output_detected_path(self):
        return f"{self.OUTPUT_DIR_DETECTED}/{self.filename}.tif"
    
    @property
    def output_segmented_path(self):
        return f"{self.OUTPUT_DIR_SEGMENTED}/{self.filename}.tif"
    
    def output_histogram_path(self, z: int):
        return f"{self.OUTPUT_DIR_HISTOGRAM}/{self.filename}-{'full' if z == None else z}.png"