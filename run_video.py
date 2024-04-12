import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

from RoiMultiClass import ComposeROI
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import global_params_variables
from process_roi import FrameProcessor

params = global_params_variables.ParamsDict()

output_image_path = params.get_value('output_image_path')
roi_comp = ComposeROI(params.get_all_items())

parser = argparse.ArgumentParser()
parser.add_argument('--video-path', type=str)
parser.add_argument('--outdir', type=str, default='./vis_video_depth')
parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])

args = parser.parse_args()

if os.path.isfile(args.video_path):
    if args.video_path.endswith('txt'):
        with open(args.video_path, 'r') as f:
            lines = f.read().splitlines()
    else:
        filenames = [args.video_path]
else:
    filenames = os.listdir(args.video_path)
    filenames = [os.path.join(args.video_path, filename) for filename in filenames if not filename.startswith('.')]
    filenames.sort()

os.makedirs(args.outdir, exist_ok=True)


def run_stream(filename):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(args.encoder)).to(
        DEVICE).eval()

    total_params = sum(param.numel() for param in depth_anything.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))

    raw_video = cv2.VideoCapture(filename)
    frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))

    filename = os.path.basename(filename)
    output_path = os.path.join(args.outdir, filename[:filename.rfind('.')] + '_video_depth.mp4')
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (frame_width, frame_height))

    _, prev_frame = raw_video.read()

    roi_frame = FrameProcessor(roi_comp, output_image_path)

    while raw_video.isOpened():
        ret, raw_frame = raw_video.read()
        if not ret:
            break

        if prev_frame is None:
            prev_frame = raw_frame

        ts = raw_video.get(cv2.CAP_PROP_POS_MSEC)
        combined_frame = roi_frame.process_frame(raw_frame, prev_frame, ts)
        cv2.imshow('Filtered Frame ', combined_frame)
        out.write(combined_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    raw_video.release()
    out.release()


if __name__ == '__main__':
    filename = params.get_value('input_video_path')

    run_stream(filename)
