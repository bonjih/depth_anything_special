import numpy as np
import cv2

size = 60
thresh = 2


def detect_blur_fft(frame):
    """
    dust is treated as a blured image
    :param frame:
    :param size: size of the square filter window to remove the center fgs from the FT image
    :param thresh: to determine whether an image is blurry or not.
    :return:
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    (h, w) = gray.shape
    (cX, cY) = (w // 2, h // 2)
    # cropped before performing the FFT to reduce the size of fft computation
    # so the entire image is not passed though the fft
    cropped_image = gray[cY - size:cY + size, cX - size:cX + size]
    fft = np.fft.fft2(cropped_image)
    fftShift = np.fft.fftshift(fft)
    # zeroing out a smaller region within the shifted FFT spectrum, removing noise
    fftShift[size - 10:size + 10, size - 10:size + 10] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)

    magnitude = 20 * np.log(np.abs(recon))
    m = np.mean(magnitude)

    return m, m <= thresh
