import math
import cv2
import numpy as np

def draw_kps_image(height, width, kps, color_list=[(255, 0, 0), (0, 255, 0), (0, 0, 255)], kps_type='v'):

    if kps_type == 'v':
        canvas = draw_v_kps_image(height, width, kps, color_list)
    else:
        raise NotImplementedError(f'kps_type {kps_type} not implemented')

    return canvas


def draw_v_kps_image(height, width, kps, color_list):
    stick_width = 4
    limb_seq = np.array([[0, 2], [1, 2]])
    kps = np.array(kps)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(len(limb_seq)):
        index = limb_seq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = int(math.degrees(math.atan2(y[0] - y[1], x[0] - x[1])))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stick_width), angle, 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in color])

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        cv2.circle(canvas, (int(x), int(y)), 4, color, -1)

    return canvas
