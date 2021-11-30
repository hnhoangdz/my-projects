import numpy as np
from cv2 import imread,resize,cvtColor,COLOR_BGR2RGB

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

def _imread(img_path):
    img = imread(img_path)
    if img[-1] == 3:
        img = cvtColor(img, COLOR_BGR2RGB)
    return img

def _resize(img,size):
    return resize(img,size)

def to_categorical(integer_classes, num_classes=2):
    integer_classes = np.asarray(integer_classes, dtype='int')
    num_samples = integer_classes.shape[0]
    categorical = np.zeros((num_samples, num_classes),dtype='int')
    categorical[np.arange(num_samples), integer_classes] = 1
    return categorical

def main():
    ca = to_categorical([5,6,8],10)
    print(ca)
if __name__ == "__main__":
    main()