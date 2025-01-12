import cv2
import numpy as np

def preprocess_image(image, target_size=(640, 640)):
    resized = cv2.resize(image, target_size)

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    normalized = rgb.astype(float) / 255.0
    
    return normalized

def apply_augmentation(image):

    brightness = np.random.uniform(0.8, 1.2)
    augmented = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
    
    return augmented

def resize_maintain_aspect(image, target_size):

    height, width = image.shape[:2]

    aspect = width / height
    
    if aspect > 1:
        new_width = target_size[0]
        new_height = int(new_width / aspect)
    else:
        new_height = target_size[1]
        new_width = int(new_height * aspect)
        
    resized = cv2.resize(image, (new_width, new_height))
    
    return resized

def create_letterbox(image, target_size=(640, 640)):
    letterboxed = np.full((target_size[1], target_size[0], 3), 114, dtype=np.uint8)

    resized = resize_maintain_aspect(image, target_size)

    pad_height = target_size[1] - resized.shape[0]
    pad_width = target_size[0] - resized.shape[1]
    
    start_h = pad_height // 2
    start_w = pad_width // 2
    
    letterboxed[start_h:start_h+resized.shape[0], 
                start_w:start_w+resized.shape[1]] = resized
    
    return letterboxed