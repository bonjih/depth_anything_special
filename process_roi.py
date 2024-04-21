import gc
from collections import deque
import cv2
import torch
import numpy as np
import torch.nn.functional as F

from dust_detect import detect_blur_fft
from stats import time_series_analysis, plot_histogram, compute_r_squared, trend_depth
from utils import get_contours, get_centre, create_rectangle_array
from vid_lables import timestamp, dusty_labels, draw_roi_poly, centre_labels

torch.cuda.empty_cache()
PYTORCH_CUDA_ALLOC_CONF = True
deque_size = 50
depth_sequence = deque(maxlen=deque_size)


def extract_resize_roi(frame, roi_pts, target_size=(518, 518)):
    """
    Extract a region of interest (ROI) from the frame, resize it, and convert it to grayscale.

    Parameters:
        frame (np.ndarray): The original frame.
        roi_pts (list): List of points defining the region of interest (ROI).
        target_size (tuple): Target size of the ROI after resizing. Default is (100, 100).

    Returns:
        numpy.ndarray: The ROI resized and converted to grayscale.
        numpy.ndarray: The mask used to extract the ROI.
    """

    mask = np.zeros_like(frame[:, :, 0])

    cv2.fillPoly(mask, [roi_pts], (255, 255, 255))
    roi = cv2.bitwise_and(frame, frame, mask=mask)

    roi_resized = cv2.resize(roi, target_size)
    roi_image = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)

    return roi_image, mask


def create_roi_mask(frame_shape, roi_points):
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [roi_points], 255)
    return mask


class FrameProcessor:
    def __init__(self, roi_comp, output_dir):
        self.roi_comp = roi_comp
        self.prev_frames = {key: deque(maxlen=2) for key in roi_comp.rois}
        self.motion_frames = {key: 0 for key in roi_comp.rois}
        self.motion_start_frame = {key: 0 for key in roi_comp.rois}
        self.output_dir = output_dir

        def process_frame(self, raw_frame, transform, depth_anything, DEVICE, frame_height, frame_width, prev_frame, ts):
        mean, dusty = detect_blur_fft(raw_frame)
        # display labels on output video
        dusty_labels(raw_frame, mean, dusty)
        timestamp(raw_frame, ts)
        depth_sts = []
        depth_mean = []
        bridge_text = ""
        r_sqr = 0

        text_y = 50

        for roi_key in self.roi_comp.rois:
            roi = self.roi_comp.rois[roi_key]
            roi_points = roi.get_polygon_points()
            draw_roi_poly(raw_frame, roi_key, roi_points)
            roi_mask = create_roi_mask(prev_frame.shape, roi_points)

            roi_image, mask = extract_resize_roi(raw_frame, roi_points, target_size=(100, 100))
            cnts = get_contours(roi_mask, mask)
            cX, cY = get_centre(cnts)
            new_points = create_rectangle_array(cX, cY)

            # Extract ROI from the frame using the mask
            roi_image, mask = extract_resize_roi(raw_frame, roi_points, target_size=(100, 100))

            frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB) / 255.0

            frame = transform({'image': frame})['image']
            frame = torch.from_numpy(frame).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                depth = depth_anything(frame)

            depth = F.interpolate(depth[None],
                                  (frame_height, frame_width),
                                  mode='bilinear',
                                  align_corners=False)[0, 0]

            depth_sequence.append([depth, ts, roi_key])

            if len(depth_sequence) >= deque_size:
                audit_list = trend_depth(depth_sequence)
                audit_means = calculate_audit_means(audit_list)
                for roi_keys, means in audit_means.items():
                    bridge_text = f"{roi_keys}: [{means[0]:.6f} {means[1]:.6f}]"

            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.cpu().numpy().astype(np.uint8)
            depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

            if not dusty:
                bridge_color = (0, 0, 255) if "Bridge" in bridge_text else (0, 255, 0)
                cv2.putText(raw_frame, bridge_text, (10, text_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bridge_color,
                            1)

            depth_resized = cv2.resize(depth_color, (raw_frame.shape[1], raw_frame.shape[0]))
            resized_mask = cv2.resize(mask, (raw_frame.shape[1], raw_frame.shape[0]))

            # Update frame with depth map for ROI
            raw_frame[resized_mask != 0] = depth_resized[resized_mask != 0]
            # centre_labels(raw_frame, roi_key, cX, cY)

            # del transform
            del depth
            gc.collect()
            text_y += 30

        return raw_frame
