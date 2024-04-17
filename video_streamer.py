import cv2

from process_roi import FrameProcessor
import transform_s


class VideoProcessor:
    def __init__(self, video_path, output_video_path, output_image_path, roi_comp, is_watching=False,
                 is_save_video=False,
                 offset=0):
        self.video_path = video_path
        self.output_video_path = output_video_path
        self.output_image_path = output_image_path
        self.roi_comp = roi_comp
        self.is_watching = is_watching
        self.is_save_video = is_save_video
        self.offset = offset

    def process_video(self):
        out = None
        frame_mask = None
        num_frames = 100
        frame_height = 0
        frame_width = 0

        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, self.offset * 1.0e3)

        if self.is_save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(self.output_video_path, fourcc, fps, (frame_width, frame_height))

        _, prev_frame = cap.read()

        roi_frame = FrameProcessor(self.roi_comp, self.output_image_path)
        transform, depth_anything, DEVICE = transform_s.transform()

        while cap.isOpened():
            ret, frame = cap.read()
            frame_number = 0
            if not ret:
                break

            if prev_frame is None and frame_mask is None:
                prev_frame = frame
                frame_mask = frame

            ts = cap.get(cv2.CAP_PROP_POS_MSEC)

            if frame_number % num_frames == 0:
                frame_mask = roi_frame.process_frame(frame, transform, depth_anything, DEVICE, frame_height,
                                                     frame_width, prev_frame, ts)

                frame_number += 1

            if self.is_watching:
                cv2.imshow('Filtered Frame ', frame_mask)

            if self.is_save_video:
                out.write(frame_mask)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        if self.is_save_video:
            out.release()
        cv2.destroyAllWindows()

    def start(self):
        self.process_video()
