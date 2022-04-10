from model import unet
from data.data import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import preprocessing
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

model = unet.UNet((None, None, 3), 22)
model.load_weights('unet_membrane.hdf5')

(train, validation) = path.path()

show_image(train[10], model)