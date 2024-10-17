import json
import random

import numpy as np
import torch
import torchvision.io
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append('../')
from datasets.utils import draw_kps_image


def get_kps_image(target_image, face_info, kps_type='v'):
    # return BGR image Tensor (C, H, W)
    kps = face_info['kps'][:3]
    _, height, width = target_image.shape
    lm_frame = draw_kps_image(int(height), int(width), kps, kps_type=kps_type)
    kps_tensor = torch.from_numpy(lm_frame)
    kps_tensor = kps_tensor.permute(2, 0, 1)
    return kps_tensor


class TalkingFaceVideo(Dataset):
    def __init__(
            self,
            image_size=(512, 512),
            image_scale=(1.0, 1.0),
            image_ratio=(0.9, 1.0),
            meta_paths=None,
            flip_rate=0.0,
            sample_rate=1,
            num_frames=10,
            reference_margin=30,
            num_padding_audio_frames=2,
            standard_audio_fps=16000,
            vae_scale_rate=8,
            audio_embeddings_interpolation_mode: str = 'linear',
            kps_type='v',
    ):
        super().__init__()

        self.image_size = image_size
        self.flip_rate = flip_rate
        self.sample_rate = sample_rate
        self.num_frames = num_frames
        self.reference_margin = reference_margin
        self.num_padding_audio_frames = num_padding_audio_frames
        self.standard_audio_fps = standard_audio_fps
        self.vae_scale_rate = vae_scale_rate
        self.audio_embeddings_interpolation_mode = audio_embeddings_interpolation_mode
        self.kps_type = kps_type

        self.videos_info = []
        for meta_path in meta_paths:
            obj = json.load(open(meta_path, "r"))
            self.videos_info.extend(obj)

        self.img_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                image_size,
                scale=image_scale,
                ratio=image_ratio,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
        ])

    def get_audio_frame_embeddings(self, audio_embeddings, frame_ids, video_len):
        # The length of the input audio embeddings is between video_len and 2*video_len
        # 1. interpolate the input audio embeddings into the embeddings with length of 2*vid_len
        audio_embeddings = torch.nn.functional.interpolate(
            audio_embeddings.permute(1, 2, 0),
            size=2 * video_len,
            mode=self.audio_embeddings_interpolation_mode,
        )[0, :, :].permute(1, 0)  # [2*vid_len, dim]

        # 2. pad zeros to the head and tail of embeddings. NOTE: padding double because of interpolation of 2*vid_len
        audio_embeddings = torch.cat([
            torch.zeros((2 * self.num_padding_audio_frames, audio_embeddings.shape[-1])),
            audio_embeddings,
            torch.zeros((2 * self.num_padding_audio_frames, audio_embeddings.shape[-1])),
        ], dim=0)

        # 3. select a sequence of audio embeddings to correspond to one video frame.
        audio_frame_embeddings = []
        for frame_idx in frame_ids:
            # Because of zero padding at the head of audio embeddings, the start sample is frame_idx directly.
            start_sample = frame_idx
            end_sample = frame_idx + 2 * self.num_padding_audio_frames
            audio_frame_embeddings.append(audio_embeddings[2 * start_sample:2 * (end_sample + 1), :])
        audio_frame_embeddings = torch.stack(audio_frame_embeddings, dim=0)

        return audio_frame_embeddings

    @staticmethod
    def get_face_mask(target_image, face_info):
        face_mask_image = torch.zeros_like(target_image)

        bbox = face_info['bbox']
        x1, y1, x2, y2 = bbox
        face_mask_image[:, int(y1):int(y2) + 1, int(x1):int(x2) + 1] = 255

        return face_mask_image

    @staticmethod
    def get_lip_mask(target_image, face_info, scale=2.0):
        lip_mask_image = torch.zeros_like(target_image)

        lip_landmarks = face_info['landmark_2d_106'][52:72]
        x1 = int(min(lip_landmarks[:, 0]))
        x2 = int(max(lip_landmarks[:, 0]))
        y1 = int(min(lip_landmarks[:, 1]))
        y2 = int(max(lip_landmarks[:, 1]))
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        x1 = cx - (w / 2) * scale
        x2 = cx + (w / 2) * scale
        y1 = cy - (h / 2) * scale
        y2 = cy + (h / 2) * scale
        lip_mask_image[:, int(y1):int(y2) + 1, int(x1):int(x2) + 1] = 255

        return lip_mask_image

    def process_reference_image(self, reference_image, do_flip, rand_state):
        reference_image = transforms_f.to_pil_image(reference_image)
        reference_image = self.augmentation(reference_image, self.img_transform, rand_state)
        if do_flip:
            reference_image = transforms_f.hflip(reference_image)
        reference_image = transforms_f.to_tensor(reference_image)
        reference_image = transforms_f.normalize(reference_image, mean=[.5], std=[.5])
        return reference_image

    def process_target_images(self, target_images, do_flip, rand_state):
        processed_target_images = []
        for target_image in target_images:
            target_image = self.process_reference_image(target_image, do_flip, rand_state)
            processed_target_images.append(target_image)
        target_images = torch.stack(processed_target_images, dim=0)  # [num_frames, 3, h, w]
        target_images = target_images.permute(1, 0, 2, 3)  # [3, num_frames, h, w]
        return target_images

    def process_kps_images(self, kps_images, do_flip, rand_state):
        processed_kps_images = []
        for kps_image in kps_images:
            kps_image = transforms_f.to_pil_image(kps_image)
            kps_image = self.augmentation(kps_image, self.img_transform, rand_state)
            if do_flip:
                kps_image = transforms_f.hflip(kps_image)
            kps_image = transforms_f.to_tensor(kps_image)
            if do_flip:
                # an easy implementation of flipping for kps images
                kps_image = torch.stack([kps_image[1], kps_image[0], kps_image[2]], dim=0)  # RGB -> GRB
            processed_kps_images.append(kps_image)
        kps_images = torch.stack(processed_kps_images, dim=0)  # [num_frames, 3, h, w]
        kps_images = kps_images.permute(1, 0, 2, 3)  # [3, num_frames, h, w]
        return kps_images

    def process_masks(self, masks, do_flip, rand_state):
        processed_masks = []
        for mask in masks:
            mask = transforms_f.to_pil_image(mask)
            mask = self.augmentation(mask, self.img_transform, rand_state)
            mask = transforms_f.resize(mask, size=[
                self.image_size[0] // self.vae_scale_rate,
                self.image_size[1] // self.vae_scale_rate
            ])
            if do_flip:
                mask = transforms_f.hflip(mask)
            mask = transforms_f.to_tensor(mask)
            mask = mask[0, ...]
            processed_masks.append(mask)
        masks = torch.stack(processed_masks, dim=0)  # [num_frames, h, w]
        masks = masks.unsqueeze(dim=0)  # [1, num_frames, h, w]
        return masks

    @staticmethod
    def augmentation(image, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(image)

    def __getitem__(self, index):
        flag = True
        while flag:
            video_info = dict(self.videos_info[index])
            video_path = video_info["video"]
            face_info_path = video_info["face_info"]
            audio_embeddings_path = video_info["audio_embeds"]

            video_frames, audio_waveform, meta_info = torchvision.io.read_video(
                video_path,
                pts_unit='sec',
                output_format="TCHW",
            )
            face_info = torch.load(face_info_path)

            video_len, aud_len = video_frames.shape[0], audio_waveform.shape[1]

            if video_len < self.num_frames:
                print(f'The video_len of {video_path} is {video_len}, which is less than num_frames {self.num_frames}. '
                      f'NOW SKIP IT!')
                index += 1
                continue

            clip_video_len = min(video_len, (self.num_frames - 1) * self.sample_rate + 1)
            start_idx = random.randint(0, video_len - clip_video_len)
            batch_ids = np.linspace(start_idx, start_idx + clip_video_len - 1, self.num_frames, dtype=int).tolist()

            left_max_reference_idx = min(batch_ids) - self.reference_margin - 1
            right_min_reference_idx = max(batch_ids) + self.reference_margin + 1
            if left_max_reference_idx < 0 and right_min_reference_idx > video_len:
                # print(f'There is no space to select a reference image in {video_path}, '
                #       f'because the maximum left reference index is {left_max_reference_idx}, '
                #       f'the minimum right reference index is {right_min_reference_idx}. '
                #       f'Both of them are not satisfied the condition: '
                #       f'(1) the maximum left reference index is bigger than 0; '
                #       f'(2) the minimum right reference index is smaller than video_len (it is {video_len} here). '
                #       f'NOW SKIP IT!')
                index += 1
                continue

            reference_idx_range = list(range(video_len))
            remove_ids = np.arange(left_max_reference_idx + 1, right_min_reference_idx - 1, dtype=int).tolist()

            for remove_idx in remove_ids:
                if remove_idx not in reference_idx_range:
                    continue
                reference_idx_range.remove(remove_idx)

            reference_idx = random.choice(reference_idx_range)
            reference_image = video_frames[reference_idx, ...]

            audio_embeddings = torch.load(audio_embeddings_path, map_location='cpu')
            audio_embeddings = audio_embeddings['global_embeds'].float().detach()  # [num_embeds, 1, dim]

            target_images, kps_images, face_masks, lip_masks = [], [], [], []
            for frame_idx in batch_ids:
                target_image = video_frames[frame_idx, ...]

                # kps_image = kps_frames[frame_idx, ...]
                kps_image = get_kps_image(target_image, face_info=face_info[frame_idx][0], kps_type=self.kps_type)

                face_mask = self.get_face_mask(target_image, face_info=face_info[frame_idx][0])
                lip_mask = self.get_lip_mask(target_image, face_info=face_info[frame_idx][0], scale=2.0)

                target_images.append(target_image)
                kps_images.append(kps_image)
                face_masks.append(face_mask)
                lip_masks.append(lip_mask)

            audio_frame_embeddings = self.get_audio_frame_embeddings(audio_embeddings, batch_ids, video_len)

            transform_rand_state = torch.get_rng_state()
            do_flip = random.random() < self.flip_rate

            reference_image = self.process_reference_image(reference_image, do_flip, transform_rand_state)
            target_images = self.process_target_images(target_images, do_flip, transform_rand_state)
            kps_images = self.process_kps_images(kps_images, do_flip, transform_rand_state)
            face_masks = self.process_masks(face_masks, do_flip, transform_rand_state)
            lip_masks = self.process_masks(lip_masks, do_flip, transform_rand_state)

            sample = dict(
                reference_image=reference_image,
                target_images=target_images,
                kps_images=kps_images,
                audio_frame_embeddings=audio_frame_embeddings,
                face_masks=face_masks,
                lip_masks=lip_masks,
            )
            return sample

    def __len__(self):
        return len(self.videos_info)


if __name__ == "__main__":
    from torchvision.transforms import ToPILImage

    dataset = TalkingFaceVideo(
        image_size=(512, 512),
        image_scale=(1.0, 1.0),
        image_ratio=(0.9, 1.0),
        meta_paths=[
            "hdtf_3kps.json",
            "p1.json",
            "p2.json",
            "p3.json",
        ],
        flip_rate=0.0,
        sample_rate=1,
        num_frames=12,
        reference_margin=30,
        num_padding_audio_frames=2,
        standard_audio_fps=16000,
        vae_scale_rate=8,
        audio_embeddings_interpolation_mode='linear',
        kps_type='dot',
    )
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1)
    save_dir = "test_show"
    for i, item in enumerate(dataloader):
        for k in item.keys():
            print(k, item[k].shape)
        reference_image = (item['reference_image'] + 1.0) * 0.5
        target_images = (item['target_images'] + 1.0) * 0.5
        face_masks = item['face_masks']
        lip_masks = item['lip_masks']
        kps_images = item['kps_images']

        reference_image = torch.clamp(reference_image, 0, 1).squeeze(0)
        target_images = torch.clamp(target_images, 0, 1)
        face_masks = torch.clamp(face_masks, 0, 1)
        lip_masks = torch.clamp(lip_masks, 0, 1)
        kps_images = torch.clamp(kps_images, 0, 1)

        to_pil = ToPILImage()
        reference_image_pil = to_pil(reference_image)

        import imageio
        writer = imageio.get_writer(f'{save_dir}/video_dot_{i}.mp4', fps=24)  # fps 是帧率
        for ii in range(target_images.shape[2]):
            tgt_image = target_images[0, :, ii, :, :]
            face_mask = face_masks[0, :, ii, :, :]
            lip_mask = lip_masks[0, :, ii, :, :]
            kps_image = kps_images[0, :, ii, :, :]

            tgt_image_pil = to_pil(tgt_image)
            face_mask_pil = to_pil(face_mask)
            lip_mask_pil = to_pil(lip_mask)
            kps_image_pil = to_pil(kps_image)

            face_mask_pil = face_mask_pil.resize((512, 512))
            lip_mask_pil = lip_mask_pil.resize((512, 512))

            ref_image_show = np.array(reference_image_pil)
            face_mask_show = np.stack((np.array(face_mask_pil),)*3, axis=-1)
            lip_mask_show = np.stack((np.array(lip_mask_pil),)*3, axis=-1)
            face_mask_show = np.array(tgt_image_pil) * 0.5 + face_mask_show * 0.5
            lip_mask_show = np.array(tgt_image_pil) * 0.5 + lip_mask_show * 0.5
            ref_kps_show = np.array(tgt_image_pil) * 0.5 + np.array(kps_image_pil) * 0.5

            combined_img = np.hstack((ref_image_show, face_mask_show, lip_mask_show, ref_kps_show))

            show_array = (combined_img).astype(np.uint8)
            writer.append_data(show_array)
        writer.close()

        print()
        input()
