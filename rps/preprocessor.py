# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 13:59:02 2021

@author: hoangdh
@email: snacky0907@gmail.com

"""

from cv2 import imread,resize,cvtColor,COLOR_BGR2RGB

def _preprocess_input(x):
    x = x.astype('float32')
    x = x / 255.0
    return x

def _imread(img_path):
    img = imread(img_path)
    if img[-1] == 3:
        img = cvtColor(img, COLOR_BGR2RGB)
    return img

def _resize(img,size):
    return resize(img,size)