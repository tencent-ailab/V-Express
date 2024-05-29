import torch
import math
import pathlib

import cv2
import numpy as np
import os

from imageio_ffmpeg import get_ffmpeg_exe
from scipy.ndimage import median_filter


tensor_interpolation = None


def get_tensor_interpolation_method():
    return tensor_interpolation


def set_tensor_interpolation_method(is_slerp):
    global tensor_interpolation
    tensor_interpolation = slerp if is_slerp else linear


def linear(v1, v2, t):
    return (1.0 - t) * v1 + t * v2


def slerp(
        v0: torch.Tensor, v1: torch.Tensor, t: float, DOT_THRESHOLD: float = 0.9995
) -> torch.Tensor:
    u0 = v0 / v0.norm()
    u1 = v1 / v1.norm()
    dot = (u0 * u1).sum()
    if dot.abs() > DOT_THRESHOLD:
        # logger.info(f'warning: v0 and v1 close to parallel, using linear interpolation instead.')
        return (1.0 - t) * v0 + t * v1
    omega = dot.acos()
    return (((1.0 - t) * omega).sin() * v0 + (t * omega).sin() * v1) / omega.sin()


def draw_kps_image(image, kps, color_list=[(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
    stick_width = 4
    limb_seq = np.array([[0, 2], [1, 2]])
    kps = np.array(kps)

    canvas = image

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


def save_video(video_tensor, audio_path, output_path, fps=30.0):
    pathlib.Path(output_path).parent.mkdir(exist_ok=True, parents=True)

    video_tensor = video_tensor[0, ...]
    _, num_frames, height, width = video_tensor.shape

    video_tensor = video_tensor.permute(1, 2, 3, 0)
    video_np = (video_tensor * 255).numpy().astype(np.uint8)
    video_np_filtered = median_filter(video_np, size=(3, 3, 3, 1))

    output_name = pathlib.Path(output_path).stem
    temp_output_path = output_path.replace(output_name, output_name + '-temp')
    video_writer = cv2.VideoWriter(temp_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for i in range(num_frames):
        frame_image = video_np_filtered[i]
        frame_image = cv2.cvtColor(frame_image, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_image)
    video_writer.release()

    cmd = (f'{get_ffmpeg_exe()} -i "{temp_output_path}" -i "{audio_path}" '
           f'-map 0:v -map 1:a -c:v h264 -shortest -y "{output_path}" -loglevel quiet')
    os.system(cmd)
    os.system(f'rm -rf "{temp_output_path}"')


def compute_dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def compute_ratio(kps):
    l_eye_x, l_eye_y = kps[0][0], kps[0][1]
    r_eye_x, r_eye_y = kps[1][0], kps[1][1]
    nose_x, nose_y = kps[2][0], kps[2][1]
    d_left = compute_dist(l_eye_x, l_eye_y, nose_x, nose_y)
    d_right = compute_dist(r_eye_x, r_eye_y, nose_x, nose_y)
    ratio = d_left / (d_right + 1e-6)
    return ratio


def point_to_line_dist(point, line_points):
    point = np.array(point)
    line_points = np.array(line_points)
    line_vec = line_points[1] - line_points[0]
    point_vec = point - line_points[0]
    line_norm = line_vec / np.sqrt(np.sum(line_vec ** 2))
    point_vec_scaled = point_vec * 1.0 / np.sqrt(np.sum(line_vec ** 2))
    t = np.dot(line_norm, point_vec_scaled)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = line_points[0] + t * line_vec
    dist = np.sqrt(np.sum((point - nearest) ** 2))
    return dist


def get_face_size(kps):
    # 0: left eye, 1: right eye, 2: nose
    A = kps[0, :]
    B = kps[1, :]
    C = kps[2, :]

    AB_dist = math.sqrt((A[0] - B[0])**2 + (A[1] - B[1])**2)
    C_AB_dist = point_to_line_dist(C, [A, B])
    return AB_dist, C_AB_dist


def get_rescale_params(kps_ref, kps_target):
    kps_ref = np.array(kps_ref)
    kps_target = np.array(kps_target)

    ref_AB_dist, ref_C_AB_dist = get_face_size(kps_ref)
    target_AB_dist, target_C_AB_dist = get_face_size(kps_target)

    scale_width = ref_AB_dist / target_AB_dist
    scale_height = ref_C_AB_dist / target_C_AB_dist

    return scale_width, scale_height


def retarget_kps(ref_kps, tgt_kps_list, only_offset=True):
    ref_kps = np.array(ref_kps)
    tgt_kps_list = np.array(tgt_kps_list)

    ref_ratio = compute_ratio(ref_kps)

    ratio_delta = 10000
    selected_tgt_kps_idx = None
    for idx, tgt_kps in enumerate(tgt_kps_list):
        tgt_ratio = compute_ratio(tgt_kps)
        if math.fabs(tgt_ratio - ref_ratio) < ratio_delta:
            selected_tgt_kps_idx = idx
            ratio_delta = tgt_ratio

    scale_width, scale_height = get_rescale_params(
        kps_ref=ref_kps,
        kps_target=tgt_kps_list[selected_tgt_kps_idx],
    )

    rescaled_tgt_kps_list = np.array(tgt_kps_list)
    rescaled_tgt_kps_list[:, :, 0] *= scale_width
    rescaled_tgt_kps_list[:, :, 1] *= scale_height

    if only_offset:
        nose_offset = rescaled_tgt_kps_list[:, 2, :] - rescaled_tgt_kps_list[0, 2, :]
        nose_offset = nose_offset[:, np.newaxis, :]
        ref_kps_repeat = np.tile(ref_kps, (tgt_kps_list.shape[0], 1, 1))

        ref_kps_repeat[:, :, :] -= (nose_offset / 2.0)
        rescaled_tgt_kps_list = ref_kps_repeat
    else:
        nose_offset_x = rescaled_tgt_kps_list[0, 2, 0] - ref_kps[2][0]
        nose_offset_y = rescaled_tgt_kps_list[0, 2, 1] - ref_kps[2][1]

        rescaled_tgt_kps_list[:, :, 0] -= nose_offset_x
        rescaled_tgt_kps_list[:, :, 1] -= nose_offset_y

    return rescaled_tgt_kps_list
