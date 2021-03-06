import cv2
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

# Path
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_alt2.xml'
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.67-0.65.h5'
gender_model_path = '../trained_models/gender_models/gender_mini_XCEPTION.05-0.96.h5'
emotion_labels = get_labels('fer2013')
gender_labels = get_labels('gender')
font = cv2.FONT_HERSHEY_SIMPLEX
image_path = 'test_girl.jpg'

# Draw bounding boxes
gender_offsets = (30, 60)
gender_offsets = (10, 10)
emotion_offsets = (20, 40)
emotion_offsets = (0, 0)

# Loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
gender_classifier = load_model(gender_model_path, compile=False)

# Input model shapes 
emotion_target_size = emotion_classifier.input_shape[1:3]
gender_target_size = gender_classifier.input_shape[1:3]

# Load img
img = cv2.imread(image_path)
rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray_image = np.squeeze(gray_image)
# gray_image = gray_image.astype('uint8')

# Detection
faces = detect_faces(face_detection, gray_image)
print(faces)
# plt.imshow(gray_image,cmap='gray')
# plt.show()

for face_coordinates in faces:
    x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
    rgb_face = rgb_image[y1:y2, x1:x2]

    x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
    gray_face = gray_image[y1:y2, x1:x2]
    
    # Resize to required image
    try:
        rgb_face = cv2.resize(rgb_face, (gender_target_size))
        gray_face = cv2.resize(gray_face, (emotion_target_size))
    except:
        continue
    
    # Predict gender
    rgb_face = preprocess_input(rgb_face,False)
    rgb_face = np.expand_dims(rgb_face, 0)
    gender_prediction = gender_classifier.predict(rgb_face)
    gender_label_arg = np.argmax(gender_prediction)
    gender_text = gender_labels[gender_label_arg]
    print(gender_text)
    # Predict gender
    gray_face = preprocess_input(gray_face, True)
    gray_face = np.expand_dims(gray_face, 0)
    gray_face = np.expand_dims(gray_face, -1)
    emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
    emotion_text = emotion_labels[emotion_label_arg]
    print(emotion_text)
    if gender_text == gender_labels[0]:
        color = (0, 0, 255)
    else:
        color = (255, 0, 0)

    draw_bounding_box(face_coordinates, rgb_image, color)
    draw_text(face_coordinates, rgb_image, gender_text, color, 0, -20, 1, 2)
    draw_text(face_coordinates, rgb_image, emotion_text, color, 0, -50, 1, 2)
    
bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
cv2.imwrite('../images/predicted_test_girl_image.png', bgr_image)

