from insightface.app import FaceAnalysis
import cv2
import torch


model_root_path = '../../model_ckpts/insightface_models/'

vid_path = 'RD_Radio10_0_clip_0.mp4'
face_info_path = 'RD_Radio10_0_clip_0_face_info_test.pt'

app = FaceAnalysis(
    providers=['CUDAExecutionProvider'],
    provider_options=[{'device_id': 0}],
    root=model_root_path,
)
app.prepare(ctx_id=0, det_size=(512, 512))

frames = []
video_capture = cv2.VideoCapture(vid_path)
while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break
    frames.append(frame)

face_info = []
drop_flag = False
for frame in frames:
    faces = app.get(frame)
    if len(faces) != 1:
        drop_flag = True
        break

    if not drop_flag:
        face_info.append([{
            'bbox': face.bbox,
            'kps': face.kps,
            'det_score': face.det_score,
            'landmark_3d_68': face.landmark_3d_68,
            'pose': face.pose,
            'landmark_2d_106': face.landmark_2d_106,
            'gender': face.gender,
            'age': face.age,
            'embedding': face.embedding,
        } for face in faces])
    else:
        print('error', vid_path)

torch.save(face_info, face_info_path)
print(f'saving face info to {face_info_path}')
