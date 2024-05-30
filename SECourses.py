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

def auto_crop_image(image_path, expand_percent=0.2, crop_size=(512, 512)):
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
    
    if faces:
        # Assuming 'faces' is a dictionary of detected faces
        # Pick the first face detected
        face = list(faces.values())[0]
        x, y, w, h = face['facial_area']
        
        # Calculate the expansion in pixels
        expand_pixels_w = int(w * expand_percent)
        expand_pixels_h = int(h * expand_percent)
        
        # Calculate the new crop boundaries
        left = max(0, x - expand_pixels_w)
        top = max(0, y - expand_pixels_h)
        right = min(img.width, x + w + expand_pixels_w)
        bottom = min(img.height, y + h + expand_pixels_h)
        
        # Ensure the crop is square
        crop_width = right - left
        crop_height = bottom - top
        if crop_width > crop_height:
            diff = crop_width - crop_height
            top = max(0, top - diff // 2)
            bottom = min(img.height, bottom + diff // 2)
        elif crop_height > crop_width:
            diff = crop_height - crop_width
            left = max(0, left - diff // 2)
            right = min(img.width, right + diff // 2)
        
        # Crop the image with the expanded boundaries
        cropped_img = img.crop((left, top, right, bottom))
        
        # Resize the cropped image to the desired size (512x512 by default) with best quality
        resized_img = cropped_img.resize(crop_size, resample=Image.LANCZOS)
        
        # Save the resized image as PNG
        resized_img_path = image_path.rsplit('.', 1)[0] + '.png'
        resized_img.save(image_path, format='PNG')


def generate_output_video(reference_image_path, audio_path, kps_path, output_path, retarget_strategy, num_inference_steps, reference_attention_weight, audio_attention_weight, auto_crop, crop_width, crop_height):
    print("auto cropping...")
    if auto_crop:
        auto_crop_image(reference_image_path, crop_size=(crop_width, crop_height))
    
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
def process_input(reference_image, target_input, output_path, retarget_strategy, num_inference_steps, reference_attention_weight, audio_attention_weight, auto_crop, crop_width, crop_height):
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
    
    if not output_path:
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_file_name = f"{input_file_name}_result_"
        output_file_name=sanitize_folder_name(output_file_name)
        output_file_ext = ".mp4"
        output_file_count = 1
        while os.path.exists(os.path.join(output_dir, f"{output_file_name}{output_file_count:04d}{output_file_ext}")):
            output_file_count += 1
        output_path = os.path.join(output_dir, f"{output_file_name}{output_file_count:04d}{output_file_ext}")
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    output_video_path, cropped_image_path = generate_output_video(reference_image, audio_path, kps_path, output_path, retarget_strategy, num_inference_steps, reference_attention_weight, audio_attention_weight, auto_crop,crop_width,crop_height)
    
    return output_video_path, cropped_image_path

def launch_interface():
    retarget_strategies = ["fix_face", "no_retarget", "offset_retarget", "naive_retarget"]
   
    with gr.Blocks() as demo:
        gr.Markdown("# Tencent AI Lab - V-Express Image to Animation V1 : https://www.patreon.com/posts/105251204")
        with gr.Row():          
            with gr.Column():
                input_image = gr.Image(label="Reference Image", format="png", type="filepath", height=512)
                generate_button = gr.Button("Generate Talking Video")
                retarget_strategy = gr.Dropdown(retarget_strategies, label="Retarget Strategy", value="fix_face")
                inference_steps = gr.Slider(10, 90, step=1, label="Number of Inference Steps", value=30)
                reference_attention = gr.Slider(0.80, 1.1, step=0.01, label="Reference Attention Weight", value=0.95)
                audio_attention = gr.Slider(1.0, 5.0, step=0.1, label="Audio Attention Weight", value=3.0)
                auto_crop = gr.Checkbox(label="Auto Crop Image", value=True)
                with gr.Row(visible=True) as crop_size_row:
                    crop_width = gr.Number(label="Crop Width", value=512)
                    crop_height = gr.Number(label="Crop Height", value=512)
        
            with gr.Column():
                input_video = gr.File(
                    label="Target Input (Image or Video)",
                    type="filepath",
                    file_count="single",
                    file_types=[
                        ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp",  # Image extensions
                        ".mp4", ".avi", ".mov", ".wmv", ".flv", ".mkv", ".webm"    # Video extensions
                    ],
                    height=512        )
                video_output = gr.Video(visible=False)
                audio_output = gr.Audio(visible=False)
    
                input_video.change(display_media, inputs=input_video, outputs=[video_output, audio_output])

                output_path = gr.Textbox(label="Output Path (leave blank for default)")
                gr.Markdown("""

                            Retarget Strategies

                            Only target audio : fix_face

                            Input picture and target video (same person - best practice) select : no_retarget

                            Input picture and target video (different person) select : offset_retarget or naive_retarget

                            naive_retarget supposed to be better than offset_retarget

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
                output_path,
                retarget_strategy,
                inference_steps,
                reference_attention,
                audio_attention,
                auto_crop,
                crop_width,
                crop_height
            ],
            outputs=[output_video, output_image]
        )
    
    demo.queue()
    demo.launch(inbrowser=True,share=args.share)

# Run the Gradio interface
launch_interface()