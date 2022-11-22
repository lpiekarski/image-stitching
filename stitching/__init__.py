import random

import numpy as np
import cv2


# Apply projective transformation to the image
def transform(img: np.ndarray, transformation: np.ndarray):
    if transformation.shape != (3, 3):
        raise ValueError(f'Transformation should be a 3x3 matrix. Got shape {transformation.shape}')
    inverse_transformation = np.linalg.inv(transformation)

    # Find bounding box for the image after transformation
    p1 = transformation @ [0, 0, 1.]
    p1 /= p1[2]
    p2 = transformation @ [img.shape[0], 0, 1.]
    p2 /= p2[2]
    p3 = transformation @ [0, img.shape[1], 1.]
    p3 /= p3[2]
    p4 = transformation @ [img.shape[0], img.shape[1], 1.]
    p4 /= p4[2]
    bbox_min = np.floor(np.min([p1, p2, p3, p4], axis=0)).astype(np.int32)
    bbox_max = np.ceil(np.max([p1, p2, p3, p4], axis=0)).astype(np.int32)

    # Fill the transformed image
    result = np.zeros((bbox_max[0] - bbox_min[0], bbox_max[1] - bbox_min[1], img.shape[2]), dtype=np.uint8)
    for i in range(bbox_max[0] - bbox_min[0]):
        for j in range(bbox_max[1] - bbox_min[1]):
            orig_img_coords = np.matmul(inverse_transformation, [i + bbox_min[0], j + bbox_min[1], 1.])
            orig_img_coords /= orig_img_coords[2]
            orig_img_coords = np.round(orig_img_coords).astype(np.int32)  # choose nearest neighbor
            if 0 <= orig_img_coords[0] < img.shape[0] and 0 <= orig_img_coords[1] < img.shape[1]:
                result[i][j][:] = img[orig_img_coords[0]][orig_img_coords[1]][:]

    # Return image after transformation and the offset
    return result, bbox_min[:2]


def find_projective_transformation(matching_points):
    matching_points = [(np.array([pt1[0], pt1[1], 1]), np.array([pt2[0], pt2[1], 1])) for pt1, pt2 in matching_points]
    mtx = np.zeros((len(matching_points) * 2, 9))
    for idx, (source_point, destination_point) in enumerate(matching_points):
        mtx[idx * 2][:3] = source_point
        mtx[idx * 2 + 1][3:6] = source_point
        mtx[idx * 2][6:] = -1 * destination_point[0] * source_point
        mtx[idx * 2 + 1][6:] = -1 * destination_point[1] * source_point
    _, _, v = np.linalg.svd(mtx)
    return np.reshape(v[-1, :], (3, 3))


def generate_weightmask(shape):
    result = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    for i in range(shape[0]):
        for j in range(shape[1]):
            dist_to_edge = min([i, shape[0] - i, j, shape[1] - j])
            metric = min(1, 2 * dist_to_edge / min(shape[0], shape[1]))
            result[i][j] = np.array([255 * metric, 255 * metric, 255 * metric], dtype=np.uint8)
    return result


def stitch_images(img1, img2, matched_points):
    img1_weightmask = generate_weightmask(img1.shape)
    img2_weightmask = generate_weightmask(img2.shape)
    homography = find_projective_transformation(matched_points)
    img1, offset = transform(img1, homography)
    img1_weightmask, _ = transform(img1_weightmask, homography)
    bbox1_min = offset
    bbox1_max = offset + img1.shape[:2]
    bbox2_min = np.array([0, 0])
    bbox2_max = img2.shape[:2]
    stitched_bbox_min = np.min([bbox1_min, bbox2_min], axis=0)
    stitched_bbox_max = np.max([bbox1_max, bbox2_max], axis=0)
    stitched_img = np.zeros((
        stitched_bbox_max[0] - stitched_bbox_min[0],
        stitched_bbox_max[1] - stitched_bbox_min[1],
        img1.shape[2]
    ), dtype=np.uint8)

    for i in range(stitched_bbox_max[0] - stitched_bbox_min[0]):
        for j in range(stitched_bbox_max[1] - stitched_bbox_min[1]):
            coord_on_img1 = (i + stitched_bbox_min[0] - offset[0], j + stitched_bbox_min[1] - offset[1])
            coord_on_img2 = (i + stitched_bbox_min[0], j + stitched_bbox_min[1])
            in_range_for_img1 = 0 <= coord_on_img1[0] < img1.shape[0] and 0 <= coord_on_img1[1] < img1.shape[1]
            in_range_for_img2 = 0 <= coord_on_img2[0] < img2.shape[0] and 0 <= coord_on_img2[1] < img2.shape[1]
            if in_range_for_img2 and not in_range_for_img1:
                stitched_img[i][j] = img2[coord_on_img2[0]][coord_on_img2[1]]
            elif in_range_for_img1 and not in_range_for_img2:
                stitched_img[i][j] = img1[coord_on_img1[0]][coord_on_img1[1]]
            elif in_range_for_img1 and in_range_for_img2:
                weight1 = img1_weightmask[coord_on_img1[0]][coord_on_img1[1]] + 1e-8
                weight2 = img2_weightmask[coord_on_img2[0]][coord_on_img2[1]] + 1e-8
                stitched_img[i][j] = np.floor((weight2 * img2[coord_on_img2[0]][coord_on_img2[1]] + weight1 *
                                               img1[coord_on_img1[0]][coord_on_img1[1]]) / (weight1 + weight2)).astype(np.uint8)

    return stitched_img


def ransac(matching_points, n=5, k=100, t=1):
    iterations = 0
    best_point_count = 0
    best_fit = None
    t = t ** 2
    while iterations < k:
        maybe_inliers = random.sample(matching_points, n)
        maybe_model = find_projective_transformation(maybe_inliers)
        also_inliers = set()
        for point in matching_points:
            s_point, d_point = point
            s_trans = maybe_model @ (s_point[0], s_point[1], 1.)
            s_trans = (s_trans[0] / s_trans[2], s_trans[1] / s_trans[2])
            if (s_trans[0] - d_point[0]) ** 2 + (s_trans[1] - d_point[1]) ** 2 < t:
                also_inliers.add(point)
        if len(also_inliers) > best_point_count:
            best_fit = also_inliers.copy()
            best_point_count = len(also_inliers)
        iterations += 1

    return list(best_fit)


# This function is heavily inspired by an OpenCV tutorial:
# https://docs.opencv.org/4.6.0/dc/dc3/tutorial_py_matcher.html
#
# You can use it as is, you do not have to understand the insides.
# You need to pass filenames as arguments.
# You can disable the preview with visualize=False.
# lowe_ratio controls filtering of matches, increasing it
# will increase number of matches, at the cost of their quality.
#
# Return format is a list of matches, where match is a tuple of two keypoints.
# First keypoint designates coordinates on the first image.
# Second one designates the same feature on the second image.
def get_matches(filename1, filename2, visualize=True, lowe_ratio=0.6):
    # Read images from files, convert to greyscale
    img1 = cv2.imread(filename1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(filename2, cv2.IMREAD_GRAYSCALE)

    # Find the keypoints and descriptors with SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    # Ratio test as per Lowe's paper
    good_matches = []
    for i, (m, n) in enumerate(matches):
        if m.distance < lowe_ratio * n.distance:
            matchesMask[i] = [1, 0]
            good_matches.append((kp1[m.queryIdx].pt, kp2[m.trainIdx].pt))

    if visualize:
        draw_params = dict(
            matchColor=(0, 255, 0),
            singlePointColor=(0, 0, 255),
            matchesMask=matchesMask,
            flags=cv2.DrawMatchesFlags_DEFAULT,
        )
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
        cv2.imshow("vis", img3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return good_matches
