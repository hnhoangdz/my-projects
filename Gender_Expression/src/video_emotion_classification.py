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
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.67-0.65.h5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for drawing bounding box
frame_window = 10 
emotion_offsets = (20, 40) # Offsets for drawing

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

# get required emotion shape
emotion_target_size = emotion_classifier.input_shape[1:3]

# Store 
emotion_window = []

video_capture = cv2.VideoCapture(0)

while True:
    bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    
    # Detect face in grayscale  
    faces = detect_faces(face_detection, gray_image)
    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue
        
        # Normalize image [-1,1]
        gray_face = preprocess_input(gray_face, True)
        
        # Expand dims of image to dive into model
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        
        # Predict 
        emotion_prediction = emotion_classifier.predict(gray_face)
        
        # Get highest accuracy emotions
        emotion_probability = np.max(emotion_prediction)
        
        # Get label name 
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        # print(emotion_probability)
        # Get highest accuracy after 10 frames and remove result on first frame
        emotion_window.append(emotion_text)
        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue
        
        # Change color for each emotion
        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))
        color = color.astype(int)
        color = color.tolist()
        
        # Drawing bouding box and emotion in text
        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode,
                  color, 0, -45, 1, 1)
    # Display video
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()