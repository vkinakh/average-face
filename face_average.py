import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np


def read_data(path: Path) -> List[Dict]:
    """Reads the data for average face

    Args:
        path: Path to folder with images and text files with points

    Returns:
        List[Dict]: List with dicts with image, face points and image filename"""

    data_array = []

    image_ptn = '*.jpg'
    points_file_post = '.txt'
    points_num = 68

    for file in path.glob(image_ptn):
        image = cv2.imread(str(file))
        image = np.float32(image) / 255.0

        points_file = Path(str(file) + points_file_post)
        if not points_file.exists():
            continue

        points = []

        with open(points_file) as f:
            for l in f:
                x, y = l.split()
                points.append((int(x), int(y)))

        if len(points) != points_num:
            continue

        data_array.append({'image': image, 'points': points, 'file': file})
    return data_array


def similarity_transform(in_points: List, out_points: List) -> np.ndarray:
    """Computes similarity transform given two sets of two points.
    OpenCV requires 3 pairs of corresponding points.
    The third one is faked by hallucinating such point that it forms an equilateral triangle

    Args:
        in_points: First set of points

        out_points: Second set of points

    Returns:
        np.ndarray: computed similarity transform"""

    s60 = np.sin(60 * np.pi / 180)
    c60 = np.cos(60 * np.pi / 180)

    in_pts = np.copy(in_points).tolist()
    out_pts = np.copy(out_points).tolist()

    xin = c60 * (in_pts[0][0] - in_pts[1][0]) - s60 * (in_pts[0][1] - in_pts[1][1]) + in_pts[1][0]
    yin = s60 * (in_pts[0][0] - in_pts[1][0]) + c60 * (in_pts[0][1] - in_pts[1][1]) + in_pts[1][1]

    in_pts.append([np.int(xin), np.int(yin)])

    xout = c60 * (out_pts[0][0] - out_pts[1][0]) - s60 * (out_pts[0][1] - out_pts[1][1]) + out_pts[1][0]
    yout = s60 * (out_pts[0][0] - out_pts[1][0]) + c60 * (out_pts[0][1] - out_pts[1][1]) + out_pts[1][1]

    out_pts.append([np.int(xout), np.int(yout)])

    tform = cv2.estimateAffinePartial2D(np.array([in_pts]), np.array([out_pts]))
    return tform[0]


def rect_contains(rect: Tuple, point: Tuple) -> bool:
    """Checks if a point is inside a rectangle

    Args:
        rect: Top left and bottom right corners positions of rectangle

        point: Point to check

    Returns:
        bool: if a point is inside a rectangle"""

    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


def calculate_delaunay_triangles(rect: Tuple, points: np.ndarray) -> List[Tuple]:
    """Calculates Delaunay triangles for detected face landmarks

    Args:
        rect: Face rectangle

        points: Detected face landmarks

    Returns:
        List[Tuple]: calculated Delaunay triangles"""

    # Create subdiv
    subdiv = cv2.Subdiv2D(rect)

    # Insert points into subdiv
    for p in points:
        subdiv.insert((p[0], p[1]))

    # List of triangles. Each triangle is a list of 3 points ( 6 numbers )
    triangle_list = subdiv.getTriangleList()

    # Find the indices of triangles in the points array
    delaunay_triangles = []

    for t in triangle_list:
        pt = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3):
            ind = []
            for j in range(0, 3):
                for k in range(0, len(points)):
                    if abs(pt[j][0] - points[k][0]) < 1. and abs(pt[j][1] - points[k][1]) < 1.:
                        ind.append(k)
            if len(ind) == 3:
                delaunay_triangles.append((ind[0], ind[1], ind[2]))

    return delaunay_triangles


def constrain_point(p: np.ndarray, w: int, h: int):
    """Constrains point so it is within images bounds

    Args:
        p: Point to constrain
        w: Width of the image
        h: Height of the image

    Returns:
        Tuple: constrained point"""

    p = (min(max(p[0], 0), w - 1), min(max(p[1], 0), h - 1))
    return p


def apply_affine_transform(image: np.ndarray, src_tri, dst_tri, size) -> np.ndarray:
    """ Applies affine transform calculated using src_tri and dst_tri to image and output an image of size.

    Args:
        image: Image to apply affine transformation

        src_tri: First set of points to calculate affine transform

        dst_tri: Second set of points to calculate affine transform

        size: Output size of the warped image

    Returns:
        np.ndarray: Warped image"""

    # Given a pair of triangles, find the affine transform.
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))

    # Apply the Affine Transform just found to the image image
    image_warped = cv2.warpAffine(image, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT_101)

    return image_warped


def warp_triangle(img1: np.ndarray, img2: np.ndarray, t1: List, t2: List) -> np.ndarray:
    """Warps and alpha blends triangular regions from img1 and img2 to img

    Args:
        img1: Input image to warp and alpha blend

        img2: Output image to warp and alpha blend

        t1: List of triangle points

        t2: List of triangle points

    Returns:
        np.ndarray: image warped and alpha blended triangular regions"""

    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1_rect = []
    t2_rect = []
    t2_rect_int = []

    for i in range(3):
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2_rect_int.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    size = (r2[2], r2[3])

    img2_rect = apply_affine_transform(img1_rect, t1_rect, t2_rect, size)
    img2_rect = img2_rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * \
                                                     ((1.0, 1.0, 1.0) - mask)

    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2_rect


def main(args):
    path = Path(args.path)

    # Dimensions of output image
    w = 600
    h = 600

    # Read images and detected face landmarks
    data = read_data(path)

    # Eye corners
    eyecorner_dst = [(int(0.3 * w), int(h / 3)), (int(0.7 * w), int(h / 3))]

    images_norm = []
    points_norm = []

    # Boundary points for delaunay triangulation
    boundary_pts = np.array([(0, 0), (w / 2, 0), (w - 1, 0), (w - 1, h / 2), (w - 1, h - 1), (w / 2, h - 1),
                             (0, h - 1), (0, h / 2)])

    # Initialize location of average points to 0s
    points_avg = np.array([(0, 0)] * (len(data[0]['points']) + len(boundary_pts)), np.float32)
    num_images = len(data)

    # Warp images and transform face landmarks to output coordinate system,
    # and find average of transformed landmarks.
    print('Start warping images and transform face landmarks')
    for i in range(num_images):
        points1 = data[i]['points']
        img = data[i]['image']

        # Corners of the eye in input image
        eyecorner_src = [points1[36], points1[45]]

        # Compute similarity transform
        tform = similarity_transform(eyecorner_src, eyecorner_dst)

        # Apply similarity transformation
        img = cv2.warpAffine(img, tform, (w, h))

        # Apply similarity transform on points
        points2 = np.reshape(np.array(points1), (68, 1, 2))
        points = cv2.transform(points2, tform)
        points = np.float32(np.reshape(points, (68, 2)))

        # Append boundary points. Will be used in Delaunay Triangulation
        points = np.append(points, boundary_pts, axis=0)

        # Calculate location of average landmark points.
        points_avg = points_avg + points / num_images

        points_norm.append(points)
        images_norm.append(img)

    # Delaunay triangulation
    rect = (0, 0, w, h)
    dt = calculate_delaunay_triangles(rect, np.array(points_avg))

    # Output image
    output = np.zeros((h, w, 3), np.float32())

    # Warp input images to average image landmarks
    print('Start computing average face')
    for i in range(len(images_norm)):
        img = np.zeros((h, w, 3), np.float32())
        # Transform triangles one by one
        for j in range(len(dt)):
            tin = []
            tout = []

            for k in range(3):
                p_in = points_norm[i][dt[j][k]]
                p_in = constrain_point(p_in, w, h)

                p_out = points_avg[dt[j][k]]
                p_out = constrain_point(p_out, w, h)

                tin.append(p_in)
                tout.append(p_out)

            warp_triangle(images_norm[i], img, tin, tout)

        # Add image intensities for averaging
        output = output + img

    # get average
    output = output / num_images

    # Display result
    cv2.imshow('image', output)
    cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute average image from input images')
    parser.add_argument('-p', '--path', action='store', dest='path', type=str, default='./images')

    args = parser.parse_args()
    main(args)
