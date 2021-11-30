import cv2

# Load haar cascade trained model
# For face detection task
def load_detection_model(model_path):
    detection_model = cv2.CascadeClassifier(model_path)
    return detection_model

# Detect face by grayscale image
def detect_faces(detection_model,gray_image_array):
    return detection_model.detectMultiScale(gray_image_array, 1.3, 5)

# Draw bounding box after detecting face
def draw_bounding_box(face_coordinates, image_array, color):
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)
    
# Add offsets to be more flexible
def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

# Draw emotion/gender in type of text 
def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0,
                                                font_scale=2, thickness=2):
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (x + x_offset, y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)
