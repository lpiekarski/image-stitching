import cv2
import sys

from stitching import stitch_images, get_matches, ransac


def get_arg_or_default(num, default):
    if len(sys.argv) > num:
        return sys.argv[num]
    return default


# task 7
if __name__ == "__main__":
    cv2.namedWindow("Image Stitching")
    img1_filename = get_arg_or_default(1, 'examples/undistorted000.png')
    img2_filename = get_arg_or_default(2, 'examples/undistorted001.png')
    img1 = cv2.imread(img1_filename)
    img2 = cv2.imread(img2_filename)
    point_pairs = get_matches(img1_filename, img2_filename, visualize=False)
    point_pairs = [((pt1[1], pt1[0]), (pt2[1], pt2[0])) for pt1, pt2 in point_pairs]
    point_pairs = ransac(point_pairs)

    stitched_img = stitch_images(img1, img2, point_pairs)
    cv2.imshow("Image Stitching", stitched_img)
    cv2.waitKey()
