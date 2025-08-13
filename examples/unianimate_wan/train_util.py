import copy
import math

import cv2
import matplotlib
import numpy as np

eps = 0.01


def draw_bodypose(canvas, candidate, subset):
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    stickwidth = 4

    limbSeq = [
        [2, 3],
        [2, 6],
        [3, 4],
        [4, 5],
        [6, 7],
        [7, 8],
        [2, 9],
        [9, 10],
        [10, 11],
        [2, 12],
        [12, 13],
        [13, 14],
        [2, 1],
        [1, 15],
        [15, 17],
        [1, 16],
        [16, 18],
        [3, 17],
        [6, 18],
    ]

    colors = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 255, 0],
        [170, 255, 0],
        [85, 255, 0],
        [0, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 255, 255],
        [0, 170, 255],
        [0, 85, 255],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [255, 0, 255],
        [255, 0, 170],
        [255, 0, 85],
    ]

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly(
                (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
            )
            cv2.fillConvexPoly(canvas, polygon, colors[i])

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)

    return canvas


def draw_body_and_foot(canvas, candidate, subset):
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    stickwidth = 4

    limbSeq = [
        [2, 3],
        [2, 6],
        [3, 4],
        [4, 5],
        [6, 7],
        [7, 8],
        [2, 9],
        [9, 10],
        [10, 11],
        [2, 12],
        [12, 13],
        [13, 14],
        [2, 1],
        [1, 15],
        [15, 17],
        [1, 16],
        [16, 18],
        [14, 19],
        [11, 20],
    ]

    colors = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 255, 0],
        [170, 255, 0],
        [85, 255, 0],
        [0, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 255, 255],
        [0, 170, 255],
        [0, 85, 255],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [255, 0, 255],
        [255, 0, 170],
        [255, 0, 85],
        [170, 255, 255],
        [255, 255, 0],
    ]

    for i in range(19):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly(
                (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
            )
            cv2.fillConvexPoly(canvas, polygon, colors[i])

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(20):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)

    return canvas


def draw_handpose(canvas, all_hand_peaks):
    H, W, C = canvas.shape

    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [0, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [0, 9],
        [9, 10],
        [10, 11],
        [11, 12],
        [0, 13],
        [13, 14],
        [14, 15],
        [15, 16],
        [0, 17],
        [17, 18],
        [18, 19],
        [19, 20],
    ]

    for peaks in all_hand_peaks:
        peaks = np.array(peaks)

        for ie, e in enumerate(edges):
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                cv2.line(
                    canvas,
                    (x1, y1),
                    (x2, y2),
                    matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0])
                    * 255,
                    thickness=2,
                )

        for i, keyponit in enumerate(peaks):
            x, y = keyponit
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
    return canvas


def draw_facepose(canvas, all_lmks):
    H, W, C = canvas.shape
    for lmks in all_lmks:
        lmks = np.array(lmks)
        for lmk in lmks:
            x, y = lmk
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 3, (255, 255, 255), thickness=-1)
    return canvas


def draw_pose(pose, H, W):
    bodies = pose["bodies"]
    faces = pose["faces"]
    hands = pose["hands"]
    candidate = bodies["candidate"]
    subset = bodies["subset"]
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    canvas = draw_body_and_foot(canvas, candidate, subset)

    canvas = draw_handpose(canvas, hands)

    canvas_without_face = copy.deepcopy(canvas)

    canvas = draw_facepose(canvas, faces)

    return canvas_without_face, canvas


def draw_keypoints(keypoints, size):
    H, W = size
    keypoints = keypoints[None]
    keypoints[..., 0] /= float(W)
    keypoints[..., 1] /= float(H)
    bodyfoot = np.concatenate([keypoints[:, :8], keypoints[:, 9:25]], axis=1)

    nums = 1
    for i in range(nums):
        if -1 not in bodyfoot[i][18] and -1 not in bodyfoot[i][19]:
            bodyfoot[i][18] = (bodyfoot[i][18] + bodyfoot[i][19]) / 2
        else:
            bodyfoot[i][18] = np.array([-1.0, -1.0])
        if -1 not in bodyfoot[i][21] and -1 not in bodyfoot[i][22]:
            bodyfoot[i][19] = (bodyfoot[i][21] + bodyfoot[i][22]) / 2
        else:
            bodyfoot[i][19] = np.array([-1.0, -1.0])

    bodyfoot = bodyfoot[:, :20, :]
    bodyfoot = bodyfoot.reshape(nums * 20, -1)

    foot = bodyfoot[:, 18:24]

    faces = keypoints[:, 25 : 25 + 68]

    # hands = candidate[:,92:113]
    hands = np.vstack([keypoints[:, -42:-21], keypoints[:, -21:]])

    # bodies = dict(candidate=body, subset=score)
    bodies = dict(candidate=bodyfoot, subset=np.arange(20, dtype=np.float32)[None])
    pose = dict(bodies=bodies, hands=hands, faces=faces)
    return draw_pose(pose, *size)


# Index mapping from COCO-WholeBody keypoint definition to OpenPose keypoint definition
# openpose_kpts = coco_wholebody_kpts[COCO_WHOLEBODY_TO_OPENPOSE_MAPPING]
# Non-existing keypoints (e.g., neck, mid hip) are generated by regression
COCO_WHOLEBODY_TO_OPENPOSE_MAPPING = [
    # body 17, feet 6 -> body 25
    0,
    [5, 6, 0.5],
    6,
    8,
    10,
    5,
    7,
    9,
    [11, 12, 0.5],
    12,
    14,
    16,
    11,
    13,
    15,
    2,
    1,
    4,
    3,
    *(i for i in range(17, 23)),
    # face 68 -> face 70
    *(i for i in range(23, 91)),
    2,
    1,
    # hands 42 -> hands 42
    *(i for i in range(91, 133)),
]


def coco_wholebody2openpose(coco_wb_kpts: np.ndarray):
    """
    Convert keypoints in coco-wholebody definition to openpose definition.

    Arguments:
        coco_wb_kpts: 133 x 2/3 detected keypoints in coco-wholebody definition

    Return:
        137 x 2/3 converted keypoints in openpose definition
    """
    openpose_kpts = np.zeros((137, 2), dtype=np.float32)
    for i in range(len(COCO_WHOLEBODY_TO_OPENPOSE_MAPPING)):
        if isinstance(COCO_WHOLEBODY_TO_OPENPOSE_MAPPING[i], list):
            openpose_kpts[i] = (
                COCO_WHOLEBODY_TO_OPENPOSE_MAPPING[i][2]
                * coco_wb_kpts[COCO_WHOLEBODY_TO_OPENPOSE_MAPPING[i][0]]
                + (1.0 - COCO_WHOLEBODY_TO_OPENPOSE_MAPPING[i][2])
                * coco_wb_kpts[COCO_WHOLEBODY_TO_OPENPOSE_MAPPING[i][1]]
            )
        else:
            openpose_kpts[i] = coco_wb_kpts[COCO_WHOLEBODY_TO_OPENPOSE_MAPPING[i]]
    return openpose_kpts
