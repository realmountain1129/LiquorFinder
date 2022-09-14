import numpy as np
from tensorflow.keras.models import load_model
import cv2
import tensorflow as tf

model = load_model('./model/21classes.h5')
class_names = ["Mudshake", "Limoncello", "Campari", "Jose Cuervo", "Bacardi Carta Blanca", "Gilbeys Vodka", "Smirnoff", "Blue Curacao", "Jack Daniel's", "Peach Tree", "Vermouth", "Kahlua", 'Tanqueray', 'Baileys', 'Malibu', '818 Tequila', 'Bombay Sapphire', "Gordon's Gin", 'Johnnie Walker', 'Amaretto', 'Absolute']

def predict(path):
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
    img = np.array(img)/255.0
    img = img.reshape(1,224,224,3)
    
    prediction = model.predict(img)
    if np.max(prediction) < 0.5:
        return "해당되는 이미지를 찾을 수 없습니다"
    else:
        return class_names[np.argmax(prediction)]
