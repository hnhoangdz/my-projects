from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from models.models import mini_XCEPTION
from utils.datasets import DataManager
from utils.preprocessor import preprocess_input

# from utils.data_augmentation import ImageGenerator

# parameters
batch_size = 32
num_epochs = 100
do_random_crop = False
patience = 10
num_classes = 2
dataset_name = 'gender'
input_shape = (64, 64, 3)

images_path = '../datasets/Gender/'
log_file_path = '../trained_models/gender_models/gender_training.log'
trained_models_path = '../trained_models/gender_models/gender_mini_XCEPTION'

data_loader = DataManager(dataset_name,images_path,input_shape)
ground_truth_data = data_loader.get_data()
X_train,y_train,X_val,y_val = ground_truth_data
X_train,X_val = X_train/255.0,X_val/255.0
# data generator
data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)

# model parameters/compilation
model = mini_XCEPTION(input_shape, num_classes)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


# model callbacks
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                              patience=int(patience/2), verbose=1)
csv_logger = CSVLogger(log_file_path, append=False)
model_names = trained_models_path + '.{epoch:02d}-{val_accuracy:.2f}.h5'
model_checkpoint = ModelCheckpoint(model_names,
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

H =   model.fit_generator(data_generator.flow(X_train, y_train,
                            batch_size),
                            steps_per_epoch=len(X_train) / batch_size,
                            epochs=num_epochs, verbose=1, callbacks=callbacks,
                            validation_data=(X_val,y_val))