import os
import subprocess
import gradio as gr
from retinaface import RetinaFace
from PIL import Image
import filetype
from datetime import datetime
import re
import sys
import torch
import argparse

import platform, os

def open_folder():
    open_folder_path = os.path.abspath("outputs")
    if platform.system() == "Windows":
        os.startfile(open_folder_path)
    elif platform.system() == "Linux":
        os.system(f'xdg-open "{open_folder_path}"')


# Get the path to the currently activated Python executable
python_executable = sys.executable

def display_media(file):
    # Determine the type of the uploaded file using filetype
    if file is None:
       return gr.update(visible=False), gr.update(visible=False)
    kind = filetype.guess(file.name)

    if kind is None:
        return gr.update(visible=False), gr.update(visible=False)

    if kind.mime.startswith('video'):
        return gr.update(value=file.name, visible=True), gr.update(visible=False)
    elif kind.mime.startswith('audio'):
        return gr.update(visible=False), gr.update(value=file.name, visible=True)
    else:
        return gr.update(visible=False), gr.update(visible=False)


parser = argparse.ArgumentParser()
parser.add_argument("--share", type=str, default=False, help="Set to True to share the app publicly.")
args = parser.parse_args()


# Function to extract audio from video using FFmpeg
def extract_audio(video_path, audio_path):
    command = [python_executable, "-m", "ffmpeg", "-i", video_path, "-vn", "-acodec", "libmp3lame", "-q:a", "2", audio_path]
    subprocess.call(command)

# Function to convert audio to MP3 using FFmpeg
def convert_audio_to_mp3(audio_path, mp3_path):
    command = ["ffmpeg", "-i", audio_path, "-acodec", "libmp3lame", "-q:a", "2", mp3_path]
    subprocess.call(command)

# Function to generate kps sequence and audio from video
def generate_kps_sequence_and_audio(video_path, kps_sequence_save_path, audio_save_path):
    command = [python_executable, "scripts/extract_kps_sequence_and_audio.py", "--video_path", video_path, "--kps_sequence_save_path", kps_sequence_save_path, "--audio_save_path", audio_save_path]
    subprocess.call(command)

def auto_crop_image(image_path, expand_percent, crop_size=(512, 512)):
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = 'cuda'
        print("Using GPU for RetinaFace detection.")
    else:
        device = 'cpu'
        print("Using CPU for RetinaFace detection.")

    # Load image
    img = Image.open(image_path)

    # Perform face detection
    faces = RetinaFace.detect_faces(image_path)

    if not faces:
        print("No faces detected.")
        return None

    # Assuming 'faces' is a dictionary of detected faces
    # Pick the first face detected
    face = list(faces.values())[0]
    landmarks = face['landmarks']

    # Extract the landmarks
    right_eye = landmarks['right_eye']
    left_eye = landmarks['left_eye']
    right_mouth = landmarks['mouth_right']
    left_mouth = landmarks['mouth_left']

    # Calculate the distance between the eyes
    eye_distance = abs(right_eye[0] - left_eye[0])

    # Estimate the head width and height
    head_width = eye_distance * 4.5  # Increase the width multiplier
    head_height = eye_distance * 6.5  # Increase the height multiplier

    # Calculate the center point between the eyes
    eye_center_x = (right_eye[0] + left_eye[0]) // 2
    eye_center_y = (right_eye[1] + left_eye[1]) // 2

    # Calculate the top-left and bottom-right coordinates of the assumed head region
    head_left = max(0, int(eye_center_x - head_width // 2))
    head_top = max(0, int(eye_center_y - head_height // 2))  # Adjust the top coordinate
    head_right = min(img.width, int(eye_center_x + head_width // 2))
    head_bottom = min(img.height, int(eye_center_y + head_height // 2))  # Adjust the bottom coordinate

    # Save the assumed head image
    assumed_head_img = img.crop((head_left, head_top, head_right, head_bottom))
    assumed_head_img.save("assumed_head.png", format='PNG')

    # Calculate the expansion in pixels and the new dimensions
    expanded_w = int(head_width * (1 + expand_percent))
    expanded_h = int(head_height * (1 + expand_percent))

    # Calculate the top-left and bottom-right points of the expanded box
    center_x, center_y = head_left + head_width // 2, head_top + head_height // 2
    left = max(0, center_x - expanded_w // 2)
    right = min(img.width, center_x + expanded_w // 2)
    top = max(0, center_y - expanded_h // 2)
    bottom = min(img.height, center_y + expanded_h // 2)

    # Crop the image with the expanded boundaries
    cropped_img = img.crop((left, top, right, bottom))
    cropped_img.save("expanded_face.png", format='PNG')

    # Calculate the aspect ratio of the cropped image
    cropped_width, cropped_height = cropped_img.size
    aspect_ratio = cropped_width / cropped_height

    # Calculate the target dimensions based on the desired crop size
    target_width = crop_size[0]
    target_height = crop_size[1]

    # Adjust the crop to match the desired aspect ratio
    if aspect_ratio > target_width / target_height:
        # Crop from left and right
        new_width = int(cropped_height * target_width / target_height)
        left_crop = (cropped_width - new_width) // 2
        right_crop = left_crop + new_width
        top_crop = 0
        bottom_crop = cropped_height
    else:
        # Crop from top and bottom
        new_height = int(cropped_width * target_height / target_width)
        top_crop = (cropped_height - new_height) // 2
        bottom_crop = top_crop + new_height
        left_crop = 0
        right_crop = cropped_width

    # Crop the image with the adjusted boundaries
    final_cropped_img = cropped_img.crop((left_crop, top_crop, right_crop, bottom_crop))
    final_cropped_img.save("final_cropped_img.png", format='PNG')

    # Resize the cropped image to the desired size (512x512 by default) with best quality
    resized_img = final_cropped_img.resize(crop_size, resample=Image.LANCZOS)

    # Save the resized image as PNG
    resized_img.save(image_path, format='PNG')
     

def generate_output_video(reference_image_path, audio_path, kps_path, output_path, retarget_strategy, num_inference_steps, reference_attention_weight, audio_attention_weight, auto_crop, crop_width, crop_height, crop_expansion):
    print("auto cropping...")
    if auto_crop:
        auto_crop_image(reference_image_path,crop_expansion, crop_size=(crop_width, crop_height))
    
    print("starting inference...")
    command = [
        python_executable, "inference.py",
        "--reference_image_path", reference_image_path,
        "--audio_path", audio_path,
        "--kps_path", kps_path,
        "--output_path", output_path,
        "--retarget_strategy", retarget_strategy,
        "--num_inference_steps", str(num_inference_steps),
        "--reference_attention_weight", str(reference_attention_weight),
        "--audio_attention_weight", str(audio_attention_weight)
    ]
    
    with open("executed_command.txt", "w") as file:
        file.write(" ".join(command))
    
    subprocess.call(command)
    return output_path, reference_image_path

def sanitize_folder_name(name):
    # Define a regex pattern to match invalid characters for both Linux and Windows
    invalid_chars = r'[<>:"/\\|?*\x00-\x1F]'
    # Replace invalid characters with an underscore
    sanitized_name = re.sub(invalid_chars, '_', name)
    return sanitized_name

# Function to handle the input and generate the output
def process_input(reference_image, target_input, retarget_strategy, num_inference_steps, reference_attention_weight, audio_attention_weight, auto_crop, crop_width, crop_height, crop_expansion):
    # Create temp_process directory for intermediate files
    temp_process_dir = "temp_process"
    os.makedirs(temp_process_dir, exist_ok=True)
    
    input_file_name = os.path.splitext(os.path.basename(reference_image))[0]
    input_file_name=sanitize_folder_name(input_file_name)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    temp_dir = os.path.join(temp_process_dir, f"{input_file_name}_{timestamp}")
    os.makedirs(temp_dir, exist_ok=True)
    
    kind = filetype.guess(target_input)
    if not kind:
        raise ValueError("Cannot determine file type. Please provide a valid video or audio file.")
    
    mime_type = kind.mime
    
    if mime_type.startswith("video/"):  # Video input
        audio_path = os.path.join(temp_dir, "target_audio.mp3")
        kps_path = os.path.join(temp_dir, "kps.pth")
        print("generating generate_kps_sequence_and_audio...")
        generate_kps_sequence_and_audio(target_input, kps_path, audio_path)
    elif mime_type.startswith("audio/"):  # Audio input
        audio_path = target_input
        if mime_type != "audio/mpeg":
            mp3_path = os.path.join(temp_dir, "target_audio_converted.mp3")
            convert_audio_to_mp3(target_input, mp3_path)
            audio_path = mp3_path
        kps_path = ""
    else:
        raise ValueError("Unsupported file type. Please provide a video or audio file.")
    
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_file_name = f"{input_file_name}_result_"
    output_file_name=sanitize_folder_name(output_file_name)
    output_file_ext = ".mp4"
    output_file_count = 1
    while os.path.exists(os.path.join(output_dir, f"{output_file_name}{output_file_count:04d}{output_file_ext}")):
        output_file_count += 1
    output_path = os.path.join(output_dir, f"{output_file_name}{output_file_count:04d}{output_file_ext}")

    
    output_video_path, cropped_image_path = generate_output_video(reference_image, audio_path, kps_path, output_path, retarget_strategy, num_inference_steps, reference_attention_weight, audio_attention_weight, auto_crop,crop_width,crop_height, crop_expansion)
    
    return output_video_path, cropped_image_path

def launch_interface():
    retarget_strategies = ["fix_face", "no_retarget", "offset_retarget", "naive_retarget"]
   
    with gr.Blocks() as demo:
        gr.Markdown("# Tencent AI Lab - V-Express Image to Animation V2 : https://www.patreon.com/posts/105251204")
        with gr.Row():          
            with gr.Column():
                input_image = gr.Image(label="Reference Image", format="png", type="filepath", height=512)
                generate_button = gr.Button("Generate Talking Video")

                with gr.Row():
                    with gr.Column(min_width=0):
                        retarget_strategy = gr.Dropdown(retarget_strategies, label="Retarget Strategy", value="fix_face")
                    with gr.Column(min_width=0):
                        inference_steps = gr.Slider(10, 90, step=1, label="Number of Inference Steps", value=30)

                with gr.Row():
                    with gr.Column(min_width=0):
                        reference_attention = gr.Slider(0.80, 1.1, step=0.01, label="Reference Attention Weight", value=0.95)
                    with gr.Column(min_width=0):
                        audio_attention = gr.Slider(1.0, 5.0, step=0.1, label="Audio Attention Weight", value=3.0)             

                with gr.Row(visible=True) as crop_size_row:
                    with gr.Column(min_width=0):
                        auto_crop = gr.Checkbox(label="Auto Crop Image", value=True)
                    with gr.Column(min_width=0):
                        crop_expansion = gr.Slider(0.0, 1.0, step=0.01, label="Face Focus Expansion Percent", value=0.15)
                with gr.Row():
                    with gr.Column(min_width=0):
                        crop_width = gr.Number(label="Crop Width", value=512)
                    with gr.Column(min_width=0):
                        crop_height = gr.Number(label="Crop Height", value=512)
        
            with gr.Column():
                input_video = gr.File(
                    label="Target Input (Image or Video)",
                    type="filepath",
                    file_count="single",
                    file_types=[
                        ".mp4", ".avi", ".mov", ".wmv", ".flv", ".mkv", ".webm",  # Video extensions
                        ".3gp", ".m4v", ".mpg", ".mpeg", ".m2v", ".m4v", ".mts",  # More video extensions
                        ".mp3", ".wav", ".aac", ".flac", ".m4a", ".wma", ".ogg"   # Audio extensions
                    ],
                    height=512        )
                video_output = gr.Video(visible=False)
                audio_output = gr.Audio(visible=False)
    
                input_video.change(display_media, inputs=input_video, outputs=[video_output, audio_output])
                btn_open_outputs = gr.Button("Open Outputs Folder")
                btn_open_outputs.click(fn=open_folder)
                gr.Markdown("""

                            Retarget Strategies

                            Only target audio : fix_face

                            Input picture and target video (same person - best practice) select : no_retarget

                            Input picture and target video (different person) select : offset_retarget or naive_retarget

                            Please look examples in Tests folder to see which settings you like most. I feel like offset_retarget is best

                            For different types of input condition, such as reference image and target audio, we provide parameters for adjusting the role played by that condition information in the model prediction. We refer to these two parameters as reference_attention_weight and audio_attention_weight. Different parameters can be applied to achieve different effects using the following script. Through our experiments, we suggest that reference_attention_weight takes the value 0.9-1.0 and audio_attention_weight takes the value 1.0-3.0.
                            """)



            with gr.Column():
                output_video = gr.Video(label="Generated Video", height=512)
                output_image = gr.Image(label="Cropped Image")
        
        
        generate_button.click(
            fn=process_input,
            inputs=[
                input_image,
                input_video,
                retarget_strategy,
                inference_steps,
                reference_attention,
                audio_attention,
                auto_crop,
                crop_width,
                crop_height,
                crop_expansion
            ],
            outputs=[output_video, output_image]
        )
    
    demo.queue()
    demo.launch(inbrowser=True,share=args.share)

# Run the Gradio interface
launch_interface()