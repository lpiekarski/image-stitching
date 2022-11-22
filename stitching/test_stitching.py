from stitching import find_projective_transformation
import numpy as np


def test_find_projective_transformation():
    # Repeat test multiple times
    for _ in range(100_000):
        # Pick a random homography
        homography = np.random.randn(3, 3)
        pts = [np.random.randn(2) for _ in range(5)]
        pts_transformed = [homography @ [pt[0], pt[1], 1.] for pt in pts]
        pts_transformed = [(pt[0] / pt[2], pt[1] / pt[2]) for pt in pts_transformed]
        found_homography = find_projective_transformation(zip(pts, pts_transformed))
        assert np.allclose(homography / np.linalg.norm(homography), found_homography / np.linalg.norm(found_homography)) or \
               np.allclose(homography / np.linalg.norm(homography), found_homography / -np.linalg.norm(found_homography))
