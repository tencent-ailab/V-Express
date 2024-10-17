



# Training data

The training data is organized in JSON format, as shown below. Each sample includes video and audio embeddings, as well as face information corresponding to each frame.

```
[
    {
        "video": "HDTF/short_clip/RD_Radio10_0_clip_0.mp4",
        "face_info": "HDTF/new_face_info/RD_Radio10_0_clip_0.pt",
        "audio_embeds": "HDTF/short_clip_aud_embeds/RD_Radio10_0_clip_0.pt",
    },
    {
        "video": "HDTF/short_clip/RD_Radio10_0_clip_0.mp4",
        "face_info": "HDTF/new_face_info/RD_Radio10_0_clip_0.pt",
        "audio_embeds": "HDTF/short_clip_aud_embeds/RD_Radio10_0_clip_0.pt",
    }...
]
```

# Generate training data

- step1: extract audio embeds from video, `python extract_audio_embeddings.py`
- step2: extract face infos from video, `python extract_face_info.py`

# Data verification

To ensure the correctness of data extraction, we have provided one test case (RD_Radio10_0_clip_0.mp4). The features extracted using the above script can be verified for correctness through a validation script `python test_read.py`. If the feature differences are minimal like below, it can be considered that the extracted data is correct and ready for training.

<img width="961" alt="image" src="https://github.com/user-attachments/assets/1794126d-bff8-4469-8a0f-28c648a3ca4f">