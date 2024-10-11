export NO_ALBUMENTATIONS_UPDATE="1"

# model_version=stage_1
# step=40001
# dtype='fp16'
# device='cuda'
# test_stage="stage_1"

# model_version=stage_2
# step=135001
# dtype='fp16'
# device='cuda'
# test_stage="stage_2"

model_version=stage_3
step=100001
dtype='fp16'
device='cuda'
test_stage="stage_3"

denoising_unet_path="./exp_output/${model_version}/denoising_unet-${step}.pth"
reference_net_path="./exp_output/${model_version}/reference_net-${step}.pth"
v_kps_guider_path="./exp_output/${model_version}/v_kps_guider-${step}.pth"
audio_projection_path="./exp_output/${model_version}/audio_projection-${step}.pth"
motion_module_path="./exp_output/${model_version}/motion_module-${step}.pth"
audio_embeddings_type="global"


# dtype='fp16'
# device='cuda'
# test_stage="stage_3"
# step='final'

# denoising_unet_path="./model_ckpts/v-express/denoising_unet.bin"
# reference_net_path="./model_ckpts/v-express/reference_net.bin"
# v_kps_guider_path="./model_ckpts/v-express/v_kps_guider.bin"
# audio_projection_path="./model_ckpts/v-express/audio_projection.bin"
# motion_module_path="./model_ckpts/v-express/motion_module.bin"
# audio_embeddings_type="global"

retarget_strategy="no_retarget"


CUDA_VISIBLE_DEVICES=0 python inference.py \
    --reference_image_path "./test_samples/short_case/AOC/ref.jpg" \
    --audio_path "./test_samples/short_case/AOC/aud.mp3" \
    --kps_path "./test_samples/short_case/AOC/kps.pth" \
    --output_path "./output/short_case/AOC_${retarget_strategy}_${test_stage}_${step}.mp4" \
    --denoising_unet_path $denoising_unet_path \
    --reference_net_path $reference_net_path \
    --v_kps_guider_path $v_kps_guider_path \
    --audio_projection_path $audio_projection_path \
    --motion_module_path $motion_module_path \
    --retarget_strategy $retarget_strategy \
    --num_inference_steps 25 \
    --guidance_scale 2.5 \
    --audio_attention_weight 1.0 \
    --context_frames 24 \
    --dtype $dtype \
    --test_stage $test_stage
