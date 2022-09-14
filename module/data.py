import os
import platform
import random
import numpy as np
import tensorflow as tf
import cv2

IMAGE_SIZE = [224, 224]
# 데이터 로드할 때 빠르게 로드할 수 있도록하는 설정 변수
AUTOTUNE = tf.data.experimental.AUTOTUNE

BATCH_SIZE = 16
ext = ['.JPG', '.jpg', '.png', 'PNG']


def process_path(file_path, label):
    img = tf.io.read_file(file_path) # 이미지 읽기
    img = tf.image.decode_jpeg(img, channels=3) # 이미지를 uint8 tensor로 수정
    img = tf.image.convert_image_dtype(img, tf.float32) # float32 타입으로 수정
    img = tf.image.resize(img, IMAGE_SIZE) # 이미지 사이즈를 IMAGE_SIZE로 수정
    img = tf.image.per_image_standardization(img)
    return img, label

def prepare_for_training(ds):
#    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def one_hot_encoder(arr):
    temp = np.zeros((len(arr), max(arr)))
    for i, j in enumerate(arr):
        temp[i][j-1] = 1
    return temp

def get_train_data(local_path=os.getcwd()):
    train_path = os.path.join(local_path, 'data', 'train')
    slashs = '\\' if platform.system() == 'Windows' else '/'
    img_paths = []
    label_def = {}
    labels = []

    for i, (path, dir, files) in enumerate(os.walk(train_path)):
        if path.split(slashs)[-1] != 'train':
            label_def[path.split(slashs)[-1]] = i
            for filename in files:
                if os.path.splitext(filename)[-1] in ext:
                    img_paths.append(os.path.join(path, filename))
                    labels.append(i)

    img_paths = np.array(img_paths)
    labels = np.array(labels)
    ran_idx = np.arange(len(labels))
    np.random.shuffle(ran_idx)
    img_paths = img_paths[ran_idx]
    labels = labels[ran_idx]

    train_size = int(len(img_paths)*0.8)
    train_x, train_y = img_paths[:train_size], labels[:train_size]
    val_x, val_y = img_paths[train_size:], labels[train_size:]

    train_list_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    val_list_ds = tf.data.Dataset.from_tensor_slices((val_x, val_y))
    for img, label in train_list_ds.take(1):
        print("img = {}".format(img.numpy()))
        print("label = {}".format(label.numpy()))

    train_ds = train_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    val_ds = val_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    train_ds = prepare_for_training(train_ds)
    val_ds = prepare_for_training(val_ds)
    return train_ds, val_ds, label_def

def get_train_data_2(local_path=os.getcwd(), img_size=(224,224,3)):
    train_path = os.path.join(local_path, 'data', 'train')
    slashs = '\\' if platform.system() == 'Windows' else '/'
    imgs = []
    label_def = {}
    labels = []

    for i, (path, dir, files) in enumerate(os.walk(train_path)):
        if path.split(slashs)[-1] != 'train':
            label_def[path.split(slashs)[-1]] = i
            for filename in files:
                if os.path.splitext(filename)[-1] in ext:
                    if img_size[2] == 3:
                        try:
                            img = cv2.imdecode(np.fromfile(os.path.join(path, filename), np.uint8), cv2.IMREAD_COLOR)
                            imgs.append(cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_LINEAR))
                        except Exception as e:
                            print(str(e))
                    else:
                        try:
                            img = cv2.imdecode(np.fromfile(os.path.join(path, filename), np.uint8), cv2.IMREAD_GRAYSCALE)
                            imgs.append(cv2.resize(img, dsize=img_size[:2], interpolation=cv2.INTER_LINEAR))
                        except Exception as e:
                            print(str(e))
                    labels.append(i)

    imgs = np.array(imgs)/255.0
    labels = np.array(labels)
    labels = one_hot_encoder(labels)
    ran_idx = np.arange(len(imgs))
    np.random.shuffle(ran_idx)
    imgs = imgs[ran_idx]
    labels = labels[ran_idx]

    train_size = int(len(imgs)*0.8)
    train_x, train_y = imgs[:train_size], labels[:train_size]
    val_x, val_y = imgs[train_size:], labels[train_size:]
    return train_x, train_y, val_x, val_y