import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os

datapath1='Corona-Chest-XRay-Dataset'
dataset_path='dataset'

categories=os.listdir(dataset_path)
print(categories)

import csv
dataset='chestxray-dataset/metadata.csv'
df = pd.read_csv(dataset)
print(df.shape)

df.head()
print(df.head())

findings = pd.read_csv("chestxray-dataset/metadata.csv", usecols = ['finding'])
print(findings)

image_names = pd.read_csv("chestxray-dataset/metadata.csv", usecols = ['filename'])
print(image_names)


positives_index=np.concatenate((np.where(findings=='COVID-19')[0],np.where(findings=='SARS')[0]))
positive_image_names=image_names[positives_index]



findings = pd.read_csv("chestxray-dataset/metadata.csv", usecols = ['modality'])
print(findings)
image_names = pd.read_csv("chestxray-dataset/metadata.csv", usecols = ['modality'])
print(image_names)


negative_index=np.where(findings=='Normal')[0]

negative_image_names=image_names[negative_index]


for negative_image_name in negative_image_names:
    image=cv2.imread(os.path.join(datapath1,'images',negative_image_name))
    try:
        cv2.imwrite(os.path.join(dataset_path,categories[0],negative_image_name),image)
    except Exception as e:
        print(e)


negative_image_names.shape

import cv2,os

data_path='dataset'
categories=os.listdir(data_path)
labels=[i for i in range(len(categories))]

label_dict=dict(zip(categories,labels)) #empty dictionary

print(label_dict)
print(categories)
print(labels)


img_size=100
data=[]
target=[]

for category in categories:
    folder_path = os.path.join (data_path, category)
    img_names = os.listdir (folder_path)

    for img_name in img_names:
        img_path = os.path.join (folder_path, img_name)
        img = cv2.imread (img_path)

        try:
            gray = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize (gray, (img_size, img_size))
            data.append (resized)
            target.append (label_dict[category])

        except Exception as e:
            print ('Exception:', e)



import numpy as np

data=np.array(data)/255.0
data=np.reshape(data,(data.shape[0],img_size,img_size,1))
target=np.array(target)

from keras.utils import np_utils

new_target=np_utils.to_categorical(target)


np.save('data',data)
np.save('target',new_target)


import numpy as np

data=np.load('data.npy')
target=np.load('target.npy')



import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import *
import tensorflow as tf
from tensorflow.keras import layers
from keras.preprocessing import image
from keras.optimizers import SGD, RMSprop
from keras.preprocessing.image import ImageDataGenerator


model: Sequential = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
#print(model.summary())
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(tf.keras.layers.Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add((Dense(1, activation='softmax')))


model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=["accuracy"])

print(model.summary())

import pathlib
import PIL
import PIL.Image
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


data_dir = pathlib.Path("Corona-Chest-XRay-Dataset")
d1 = pathlib.Path("dataset")

image_count = len(list(data_dir.glob('*.*')))

print(image_count)

roses = list(data_dir.glob('*.png'))

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
print(train_ds)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
print(val_ds)

class_names = train_ds.class_names
print(class_names)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break



from tensorflow.keras import layers
import numpy as np
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))


AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 5

model = tf.keras.Sequential([
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=3
)

