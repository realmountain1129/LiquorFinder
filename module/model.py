import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras import models
from tensorflow.keras import layers

def Custom_model(input_shape=(224, 224, 3), output_num=1):
    BASE_model = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)

    model = models.Sequential()
    model.add(BASE_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    
    if output_num==1:
        model.add((layers.Dense(1, activation='sigmoid')))
    else:
        model.add((layers.Dense(output_num, activation='softmax')))
    
    for layer in model.layers[:-3]:
        layer.trainable = False
    
    for layer in model.layers:
        print(layer, layer.trainable)
    return model