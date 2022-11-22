import cv2
import sys

from stitching import stitch_images


def get_arg_or_default(num, default):
    if len(sys.argv) > num:
        return sys.argv[num]
    return default


if __name__ == "__main__":
    cv2.namedWindow("Image Stitching")
    img1_filename = get_arg_or_default(1, 'examples/undistorted000.png')
    img2_filename = get_arg_or_default(2, 'examples/undistorted001.png')
    points_filename = get_arg_or_default(3, 'examples/matching_points_000_001.txt')
    img1 = cv2.imread(img1_filename)
    img2 = cv2.imread(img2_filename)
    with open(points_filename, 'r') as f:
        splitlines = [line.split() for line in f.readlines()]
        point_pairs = [
            ((int(splitline[1]), int(splitline[0])),
             (int(splitline[3]), int(splitline[2])))
            for splitline in splitlines]
    stitched_img = stitch_images(img1, img2, point_pairs)
    cv2.imshow("Image Stitching", stitched_img)
    cv2.waitKey()
