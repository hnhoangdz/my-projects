import cv2
from tensorflow.keras.models import load_model
import numpy as np
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
from statistics import mode


# Path
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_alt2.xml'
gender_model_path = '../trained_models/gender_models/gender_mini_XCEPTION.05-0.96.h5'
gender_labels = get_labels('gender')

# hyper-parameters for drawing bounding box
frame_window = 10 
gender_offsets = (20, 40) # Offsets for drawing

# loading models
face_detection = load_detection_model(detection_model_path)
gender_classifier = load_model(gender_model_path, compile=False)

# get required gender shape
gender_target_size = gender_classifier.input_shape[1:3]

# Store 
gender_window = []

video_capture = cv2.VideoCapture(0)

while True:
    bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    
    # Detect face in grayscale  
    faces = detect_faces(face_detection, gray_image)
    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
        rgb_face = rgb_image[y1:y2, x1:x2]
        try:
            rgb_face = cv2.resize(rgb_face, (gender_target_size))
        except:
            continue
        
        # Normalize image [0,1]
        rgb_face = preprocess_input(rgb_face,False)
        
        # Expand dims of image to dive into model
        rgb_face = np.expand_dims(rgb_face, 0)
        
        # Predict 
        gender_prediction = gender_classifier.predict(rgb_face)
        
        # Get highest accuracy emotions
        gender_probability = np.max(gender_prediction)
        
        # Get label name 
        gender_label_arg = np.argmax(gender_prediction)
        gender_text = gender_labels[gender_label_arg]

        # Get highest accuracy after 10 frames and remove result on first frame
        gender_window.append(gender_text)
        if len(gender_window) > frame_window:
            gender_window.pop(0)
        try:
            gender_mode = mode(gender_window)
        except:
            continue
        
        # Change color for each emotion
        if gender_text == 'man':
            color = gender_probability * np.asarray((255, 0, 0))
        else:
            color = gender_probability * np.asarray((0, 255, 0))
        color = color.astype(int)
        color = color.tolist()
        
        # Drawing bouding box and emotion in text
        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, gender_mode,
                  color, 0, -45, 1, 1)
    # Display video
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()

