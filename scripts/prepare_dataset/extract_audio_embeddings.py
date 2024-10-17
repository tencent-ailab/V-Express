import torch
import torchvision
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor


def prepare_audio_embeddings(audio_waveform, audio_processor, audio_encoder, device, dtype):
    audio_waveform = audio_processor(audio_waveform, return_tensors="pt", sampling_rate=16000)['input_values']
    audio_waveform = audio_waveform.to(device, dtype)
    audio_embeddings = audio_encoder(audio_waveform).last_hidden_state  # [1, num_embeds, d]

    audio_embeddings = audio_embeddings.permute(1, 0, 2) # [num_embeds, 1, d]

    return audio_embeddings


audio_path = 'RD_Radio10_0_clip_0.mp4'
aud_embeds_path = 'RD_Radio10_0_clip_0_aud_embeds_test.pt'


device = f'cuda:0'
dtype = torch.float32

audio_encoder_path = '../../model_ckpts/wav2vec2-base-960h/'
STAN_AUD_FPS = 16000
audio_encoder = Wav2Vec2Model.from_pretrained(audio_encoder_path).to(dtype=dtype, device=device)
audio_processor = Wav2Vec2Processor.from_pretrained(audio_encoder_path)

_, audio_waveform, meta_info = torchvision.io.read_video(audio_path, pts_unit='sec')
audio_sampling_rate = meta_info['audio_fps']
print(f'Length of audio is {audio_waveform.shape[1]} with the sampling rate of {audio_sampling_rate}.')
if audio_sampling_rate != STAN_AUD_FPS:
    audio_waveform = torchaudio.functional.resample(
        audio_waveform,
        orig_freq=audio_sampling_rate,
        new_freq=STAN_AUD_FPS,
    )
audio_waveform = audio_waveform.mean(dim=0)

with torch.no_grad():
    audio_embedding = prepare_audio_embeddings(audio_waveform, audio_processor, audio_encoder, device, dtype)

torch.save({'global_embeds': audio_embedding}, aud_embeds_path)
