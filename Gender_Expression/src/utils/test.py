from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
gender_model_path = '../../trained_models/gender_models/gender_mini_XCEPTION.05-0.96.h5'
gender_classifier = load_model(gender_model_path, compile=False)
# print(dir(gender_classifier))
print(gender_classifier._set_mask_keras_history_checked)
      
# plt.plot(gender_classifier['history']['acc'])
# plt.plot(gender_classifier['history']['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()