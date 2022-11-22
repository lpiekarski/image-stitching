# Image Stitching
Although there are multiple implementations of image stitching using different libraries (OpenCV, skimage, ...) this repository contains image stitching using only `numpy` for computations and `cv2` for displaying images. Both projective transformation fitting and RANSAC are implemented using `numpy`, but the undistortion is done using `cv2` library.

1. Take two photos using `image_capture.py` script. You can capture the image with space bar. The images should have rather large overlap (around 60%)