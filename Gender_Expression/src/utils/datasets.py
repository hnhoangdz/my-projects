import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import cv2
from tensorflow.keras.utils import to_categorical
import collections
class DataManager(object):
    def __init__(self, dataset_name='fer2013',
                 dataset_path=None,image_size=(48,48)):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.image_size = image_size
        if self.dataset_path is not None:
            self.dataset_path = dataset_path
        elif self.dataset_path == 'fer2013':
            self.dataset_path = '../../datasets/Fer2013/'
        elif self.dataset_path == 'gender':
            self.dataset_path = '../../datasets/Gender/'
    
    def get_data(self):
        if self.dataset_name == 'fer2013':
            ground_truth_data = self._load_fer2013()
        elif self.dataset_name == 'gender':
            ground_truth_data = self._load_gender()
        return ground_truth_data
    def _load_gender(self):
        le = preprocessing.LabelEncoder()
        # Path
        gender_train_path = str(self.dataset_path) + "Training/gender_train_faces.npy"
        gender_train_labels_path = str(self.dataset_path) + "Training/gender_train_labels.npy"
        gender_val_path = str(self.dataset_path) + "Validation/gender_val_faces.npy"
        gender_val_labels_path = str(self.dataset_path) + "Validation/gender_val_labels.npy"
        
        # Load
        gender_train = np.load(gender_train_path)
        gender_train_labels = le.fit_transform(np.load(gender_train_labels_path))
        gender_val = np.load(gender_val_path)
        gender_val_labels = le.fit_transform(np.load(gender_val_labels_path))
        
        return gender_train.astype('float32'), to_categorical(gender_train_labels,2,'float32'),\
               gender_val.astype('float32'), to_categorical(gender_val_labels,2,'float32')
    def _load_fer2013(self):
        faces_path = str(self.dataset_path) + "fer_faces.npy"
        #print(faces_path)
        emotions_path = str(self.dataset_path) + "fer_faces_target.npy"
        fer_faces = np.load(faces_path)
        faces = []
        for face in fer_faces:
            face = cv2.resize(face.astype('uint8'), self.image_size)
            faces.append(face.astype('float32'))
        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)
        emotions = np.load(emotions_path)
        emotions = to_categorical(emotions,7,'float32')
        return faces, emotions
def get_labels(dataset_name):
    if dataset_name == 'fer2013':
        return {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                4: 'sad', 5: 'surprise', 6: 'neutral'}
    elif dataset_name == 'gender':
        return {0: 'woman', 1: 'man'}
    else:
        raise Exception('Invalid dataset name')
def get_class_to_arg(dataset_name='fer2013'):
    if dataset_name == 'fer2013':
        return {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4,
                'surprise': 5, 'neutral': 6}
    elif dataset_name == 'gender':
        return {'woman': 0, 'man': 1}
    else:
        raise Exception('Invalid dataset name')
def split_data(x, y, validation_split=.2):
    ids = np.arange(len(y))
    
    X_train, X_val, y_train, y_val, id_train, id_val = train_test_split(x, y, ids, test_size = validation_split, stratify = y)
    train_data = (X_train,y_train)
    val_data = (X_val,y_val)
    return train_data,val_data
if __name__ == '__main__':
    x = DataManager("gender",'../../datasets/Gender/')
    ground_truth_data = x.get_data()
    _,y_train,_,y_val = ground_truth_data
    from collections import Counter
    # target = df.values[:,-1]
    counter = Counter(y_train)
    for k,v in counter.items():
        per = v / len(y_train) * 100
        print('Class=%s, Count=%d, Percentage=%.3f%%' % (k, v, per))