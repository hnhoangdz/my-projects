# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 14:29:38 2021

@author: hoangdh
@email: snacky0907@gmail.com

"""
import cv2

def draw_bounding_box(hand_coordinates, image_array, color):
    x, y, w, h = hand_coordinates
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)
    

def apply_offsets(hand_coordinates, offsets):
    x, y, width, height = hand_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)


def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0,
                                                font_scale=2, thickness=2):
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (x + x_offset, y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)