import torch


a = torch.load('RD_Radio10_0_clip_0_aud_embeds.pt', map_location='cpu')
b = torch.load('RD_Radio10_0_clip_0_aud_embeds_test.pt', map_location='cpu')


print(f'audio embedings verification')
print(sum(a['global_embeds'] - b['global_embeds']))

aa = torch.load('RD_Radio10_0_clip_0_face_info.pt', map_location='cpu')
bb = torch.load('RD_Radio10_0_clip_0_face_info_test.pt', map_location='cpu')


print(f'face infos verification')
print(sum(bb[0][0]['landmark_2d_106'] - aa[0][0]['landmark_2d_106']))