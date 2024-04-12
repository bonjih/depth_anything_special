import os
from collections import deque
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Compose
from depthanything.depth_anything.dpt import DepthAnything
from dust_detect import detect_blur_fft
from vid_lables import timestamp, draw_roi_poly
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import torch.nn.functional as F


def extract_resize_roi(frame, roi_pts, target_size=(100, 100)):
    """
    Extract a region of interest (ROI) from the frame, resize it, and convert it to grayscale.

    Parameters:
        frame (numpy.ndarray): The original frame.
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


def dusty_labels(frame, mean, dusty):
    pass


class FrameProcessor:
    def __init__(self, roi_comp, output_dir):
        self.roi_comp = roi_comp
        self.prev_frames = {key: deque(maxlen=2) for key in roi_comp.rois}
        self.motion_frames = {key: 0 for key in roi_comp.rois}
        self.motion_start_frame = {key: 0 for key in roi_comp.rois}
        self.output_dir = output_dir

    def process_frame(self, frame, prev_frame, ts):
        mean, dusty = detect_blur_fft(frame)
        dusty_labels(frame, mean, dusty)
        timestamp(frame, ts)

        for roi_key in self.roi_comp.rois:
            roi = self.roi_comp.rois[roi_key]
            roi_points = roi.get_polygon_points()
            draw_roi_poly(frame, roi_key, roi_points)
            roi_mask = create_roi_mask(prev_frame.shape, roi_points)

            # Extract ROI from the frame using the mask
            roi_image, mask = extract_resize_roi(frame, roi_points, target_size=(100, 100))

            DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

            depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_vitl14').to(
                DEVICE).eval()

            total_params = sum(param.numel() for param in depth_anything.parameters())
            #print('Total parameters: {:.2f}M'.format(total_params / 1e6))

            transform = Compose([
                Resize(
                    width=1280,
                    height=720,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method='lower_bound',
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ])

            frame_dep = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB) / 255.0

            frame_dep = transform({'image': frame_dep})['image']
            frame_dep = torch.from_numpy(frame_dep).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                depth = depth_anything(frame_dep)

            depth = F.interpolate(depth[None], (100, 100), mode='bilinear', align_corners=False)[0, 0]
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

            depth = depth.cpu().numpy().astype(np.uint8)
            depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

            depth_resized = cv2.resize(depth_color, (frame.shape[1], frame.shape[0]))
            resized_mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

            # Update frame with depth map for ROI
            frame[resized_mask != 0] = depth_resized[resized_mask != 0]

        return frame
