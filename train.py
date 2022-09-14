import os
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import optimizers, losses
from model import Custom_model
from data import get_train_data, get_train_data_2
from tensorflow.keras.callbacks import EarlyStopping

input_size = (224,224,3)

def main():
#    train_ds, val_ds, label_def = get_train_data(local_path=local_path)
    """
    model = Custom_model(input_shape=input_size, output_num=len(label_def))
    print(model.summary())
    model.compile(optimizer=optimizers.Adam(),
            loss=losses.SparseCategoricalCrossentropy(),
            metrics=[metrics.Accuracy()])

    model.fit(train_ds, validation_data=val_ds, epochs=10)
    """

    train_x, train_y, val_x, val_y = get_train_data_2(local_path=local_path, img_size=input_size)
    model = Custom_model(input_shape=input_size, output_num=len(train_y[0]))
    print(model.summary())
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
            loss=losses.CategoricalCrossentropy(),
            metrics=['acc'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    his = model.fit(train_x, train_y, validation_data=(val_x, val_y), batch_size=10, epochs=100, callbacks=[early_stop])
    model.evaluate(val_x, val_y, batch_size=16, verbose=1)
    
    model.save('./model/21classes.h5')
    model.save('./model/21classes')
    print("model saved")
    
    plt.plot(his.history['acc'])
    plt.plot(his.history['val_acc'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(local_path, './acc/MobileNet21_acc.png'))
#    plt.show()
    plt.clf()
    
    plt.plot(his.history['loss'])
    plt.plot(his.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(local_path, './loss/MobileNet21_loss.png'))
#    plt.show()

    
if __name__ == "__main__":
    local_path = os.getcwd()
    main()
