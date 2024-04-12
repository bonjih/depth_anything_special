import numpy as np


class ROI:
    def __init__(self, **kwargs):
        self.points = kwargs

    def get_polygon_points(self):
        num_points = len(self.points) // 2  # get the number of points based on half the number of keys
        points = [(self.points[f'x{i}'], self.points[f'y{i}']) for i in range(1, num_points + 1)]
        return np.array(points, np.int32)


class ComposeROI:
    def __init__(self, data):
        self.rois = {}
        self.thresholds = []
        self.video_file = None

        if isinstance(data, list):
            data_dict = dict(data)
        else:
            data_dict = data

        roi_points = self.extract_roi_points(data_dict)
        for roi_key, roi_data in roi_points.items():
            roi = ROI(**roi_data)
            self.rois[roi_key] = roi  # Add ROI object to dictionary using its key

        for key, value in data_dict.items():
            if key == "thresholds":
                for thresh_key, thresh_value in value.items():
                    thresh = ROI(**thresh_value)
                    self.thresholds.append(thresh)
            elif key == "input_video_path":
                self.video_file = value

    @staticmethod
    def extract_roi_points(data):
        roi_points = data.get('roi_coords', {})
        return roi_points

    def add_roi(self, roi_key, roi):
        self.rois[roi_key] = roi

    def add_threshold(self, thresh):
        self.thresholds.append(thresh)

    def __iter__(self):
        return iter(list(self.rois.values()) + self.thresholds)
