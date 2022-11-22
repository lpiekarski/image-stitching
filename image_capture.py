import cv2
import os
from cam.camera import Camera
import numpy as np


os.makedirs('screenshots', exist_ok=True)
screenshot_idx = 0
cv2.namedWindow("Image Capture")
cam = Camera(quality=8)
image_size = (640, 480)

# Camera calibration parameters obtained in prior camera calibration task
mtx = np.array([[606.12371218, 0., 322.09712228],
                [0., 611.36767281, 254.64672476],
                [0., 0., 1.]])
dist = np.array([[0.10417522, -0.30366145, -0.0007479, -0.0006221, 0.24361686]])

rect_camera_matrix = cv2.getOptimalNewCameraMatrix(mtx, dist, image_size, 0)[0]
map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, np.eye(3), rect_camera_matrix, image_size, cv2.CV_32FC1)

while True:
    cam.keep_stream_alive()
    img = cam.get_frame()
    rect_img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
    cv2.imshow("Image Capture", rect_img)

    keypress = cv2.pollKey() & 0xFF
    if keypress == ord('q'):
        break
    elif keypress == ord(' '):
        filenames = None
        while filenames is None or any([os.path.exists(filename) for filename in filenames]):
            filenames = (
                f'screenshots/distorted{screenshot_idx:03d}.png',
                f'screenshots/undistorted{screenshot_idx:03d}.png',
            )
            screenshot_idx += 1
        cv2.imwrite(filenames[0], img)
        cv2.imwrite(filenames[1], rect_img)
