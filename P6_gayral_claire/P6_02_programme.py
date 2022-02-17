checkpoint_path = "my_model/cp.ckpt"
## Paths qnd global variables :
img_dim = (300,300,3)
n_class = 4 ## nb of dog classes
dict_races = {0: 'Siberian_husky', 1: 'soft', 2: 'standard_poodle', 3: 'Scotch_terrier'}

## 
## Load Libraries, versions in  :
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

## optimiser 
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import schedules
from tensorflow.keras.callbacks import ModelCheckpoint

## resnet 50 model :
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

def main(img):
    ## prepocess image : 
    img = cv2.resize(img, (img_dim[0],img_dim[1]), interpolation = cv2.INTER_AREA)
    img = preprocess_input(img)# special preprocess from keras
    img = img.reshape(1,img_dim[0],img_dim[1],img_dim[2])


    ## call model:
    model_base_ResNet = ResNet50(weights='imagenet', include_top=False, input_shape=img_dim)
    flat1 = Flatten()(model_base_ResNet.layers[-1].output)
    class1 = Dense(1024, activation='relu')(flat1)
    output = Dense(n_class, activation='softmax')(class1)
    model = Model(inputs=model_base_ResNet.inputs, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    ## load transfert best model weights
    model.load_weights(checkpoint_path)

    ## predict
    y_hat = model.predict(img)
    return(dict_races[np.argmax(y_hat)])

