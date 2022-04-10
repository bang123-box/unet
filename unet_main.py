from sklearn.utils import validation
from model import unet
from data.data import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import preprocessing
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import *
import numpy as np
import path
import os


os.environ['CUDA_VISIBLE_DEVICES']='1'
(train, validation) = path.path()

def acc(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.uint8)
    y_true = tf.one_hot(y_true, depth=22)
    y = 2*y_true*y_pred
    dice = tf.reduce_sum(y, axis=[1, 2, 3]) / tf.reduce_sum(tf.add(y_true, y_pred), axis=[1, 2, 3])
    dice = tf.reduce_mean(dice)
    return dice

def categorical_crossentropy(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.uint8)
    y_true = tf.cast(tf.one_hot(y_true, depth=22), dtype=tf.float32)
    sum = -y_true * tf.math.log(y_pred)
    sum = tf.reduce_sum(sum, axis=[1, 2, 3])
    return tf.reduce_mean(sum, axis=0)

model = unet.UNet((None, None, 3), 22)
#model.load_weights('unet_membrane.hdf5')
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)
model.compile(optimizer='Adam', loss=categorical_crossentropy, metrics=[acc])  # loss=tf.keras.losses.CategoricalCrossentropy())
history = model.fit(traingenertor(train), steps_per_epoch=len(train)//3, epochs=5, validation_data=traingenertor(validation),
          validation_steps=len(validation)//3, callbacks=[model_checkpoint])

