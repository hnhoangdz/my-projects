# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 11:56:30 2021

@author: hoangdh
@email: snacky0907@gmail.com

"""

import cv2
import numpy as np
import math
import mediapipe as mp
from preprocessor import _preprocess_input,_resize
from algorithms import game
import time
from inference import draw_bounding_box,apply_offsets,draw_text
from tensorflow.keras.models import load_model
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

trained_hand_path = "trained_model/weights-best-rps.h5"

hand_classifier = load_model(trained_hand_path, compile=False)
hand_target_size = hand_classifier.input_shape[1:3]
hand_offsets = (20, 40)
count = 0
frame_window = 10 
def distance(x1,y1,x2,y2):
    return math.sqrt(math.pow(x1-x2, 2)+math.pow(y1-y2,2))
hand_labels = ["Paper","Rock","Scissors"]
run_text = True
# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.7,
    max_num_hands=2,
    min_tracking_confidence=0.7) as hands:
  while cap.isOpened() :
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame")
      continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image_height, image_width, _ = image.shape
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if results.multi_hand_landmarks:   
        if len(results.multi_hand_landmarks) == 2:
            p1,p2,hand_coordinates_cache,hand_window = "","",[],{}
            for index,hand_landmarks in enumerate(results.multi_hand_landmarks):
                try:
                    # Calculate coordinate for bounding box
                    w_root = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x*image_width)
                    h_root = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y*image_height)
                    
                    w_to = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x*image_width)
                    h_to = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y*image_height)
                    
                    w_mid = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x*image_width)
                    h_mid = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x*image_height)
                    
                    if distance(w_root,h_root,w_to,h_to) < distance(w_root,h_root,w_mid,h_mid):
                        center_x = (w_root+w_mid)//2
                        center_y = (h_root+h_mid)//2
                    else:
                        center_x = (w_root+w_to)//2
                        center_y = (h_root+h_to)//2
                    center_coordinates = (center_x,center_y)
                    radius = np.sqrt(np.power((center_x-w_root),2)+np.power((center_y-h_root),2))
                    x = int(center_x-radius)
                    y = int(center_y-radius)
                    w = int(2*radius)
                    h = int(2*radius)
                    
                    # Applied to get bounding box
                    hand_coordinates = (x,y,w,h)
                    hand_coordinates_cache.append(hand_coordinates)
                    x_new,y_new,w_new,h_new = apply_offsets(hand_coordinates, hand_offsets)
                    hand_img = image[w_new:h_new,x_new:y_new]
                    hand_img = _preprocess_input(hand_img)
                    hand_img = _resize(hand_img, hand_target_size)
                    hand_img = np.expand_dims(hand_img, 0)
                    
                    # Predict 
                    hand_prediction = hand_classifier.predict(hand_img)

                    # Get label name 
                    hand_label_arg = np.argmax(hand_prediction)
                    hand_text = hand_labels[hand_label_arg]
                    
                    if index == 0:
                        p1 = hand_text
                    else:
                        p2 = hand_text
                except:
                    continue
            
            hand_window["player1"] = p1
            hand_window["player2"] = p2
            
            draw_bounding_box(hand_coordinates_cache[0], image, (255, 0, 0))
            draw_text(hand_coordinates_cache[0], image,"Player 1: "+hand_window["player1"], (255, 0, 0), 0, -45, 1, 1)
            draw_bounding_box(hand_coordinates_cache[1], image, (0, 255, 0))
            draw_text(hand_coordinates_cache[1], image,"Player 2: " +hand_window["player2"], (0, 255, 0), 0, -45, 1, 1)
            result = game(hand_window["player1"],hand_window["player2"])
            result_coordinates = (image_width//3,image_height//3,100,50)
            draw_text(result_coordinates, image,result, (0, 0, 255), 0, -45, 1, 1)
            if result:
                cv2.imwrite("result.jpg",image)
                count += 1
            if count == 20:
                break
                
            hand_window = []
    # cv2.imwrite('predicted_image.png', image[w_new:h_new,x_new:y_new])
    # mp_drawing.draw_landmarks(
    #     image,
    #     hand_landmarks,
    #     mp_hands.HAND_CONNECTIONS,
    #     mp_drawing_styles.get_default_hand_landmarks_style(),
    #     mp_drawing_styles.get_default_hand_connections_style())
        else:
            # draw_text((hand_coordinates_cache[1]), image,hand_window[1], color, 0, -45, 1, 1)
            print("The system must have 2 players")
    cv2.imshow('Hands Recognition', cv2.flip(image,1))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()