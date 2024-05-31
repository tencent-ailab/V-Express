from retinaface import RetinaFace
from PIL import Image
import torch

 

def auto_crop_image(image_path=r"F:\V_Express_V1\Material\Biden_Photo_Big.png", expand_percent=0.15, crop_size=(512, 512)):
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
    resized_img_path = image_path.rsplit('.', 1)[0] + '_cropped.png'  # Change file name to avoid overwriting
    resized_img.save("resized_img.png", format='PNG')

auto_crop_image()