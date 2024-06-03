import argparse
import os
import cv2
import torch
from insightface.app import FaceAnalysis
from imageio_ffmpeg import get_ffmpeg_exe
import time
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--video_path', type=str, default='')
parser.add_argument('--kps_sequence_save_path', type=str, default='')
parser.add_argument('--audio_save_path', type=str, default='')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--insightface_model_path', type=str, default='./model_ckpts/insightface_models/')
parser.add_argument('--height', type=int, default=512)
parser.add_argument('--width', type=int, default=512)
args = parser.parse_args()

# Convert paths to absolute paths
args.video_path = os.path.abspath(args.video_path)
args.kps_sequence_save_path = os.path.abspath(args.kps_sequence_save_path)
args.audio_save_path = os.path.abspath(args.audio_save_path)
args.insightface_model_path = os.path.abspath(args.insightface_model_path)

app = FaceAnalysis(
    providers=['CUDAExecutionProvider' if args.device == 'cuda' else 'CPUExecutionProvider'],
    provider_options=[{'device_id': args.gpu_id}] if args.device == 'cuda' else [],
    root=args.insightface_model_path,
)
app.prepare(ctx_id=0, det_size=(args.height, args.width))

# Use subprocess.run() with shell=True for paths with space characters
subprocess.run(f'"{get_ffmpeg_exe()}" -i "{args.video_path}" -y -vn "{args.audio_save_path}"', shell=True)

kps_sequence = []
video_capture = cv2.VideoCapture(args.video_path)
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
frame_idx = 0

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

    start_time = time.time()
    faces = app.get(frame)
    end_time = time.time()
    duration = end_time - start_time

    assert len(faces) == 1, f'There are {len(faces)} faces in the {frame_idx}-th frame. Only one face is supported.'

    kps = faces[0].kps[:3]
    kps_sequence.append(kps)
    frame_idx += 1

    processed_frames = frame_idx
    remaining_frames = total_frames - frame_idx

    print(f"Frame {frame_idx}: Face detection duration = {duration:.4f} seconds")
    print(f"Status: Processed {processed_frames} frames, {remaining_frames} frames remaining")

# Use double quotes for paths with space characters
torch.save(kps_sequence, args.kps_sequence_save_path)