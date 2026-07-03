import os
os.environ["PATH"] += os.pathsep + 'D:/Program Files/Graphviz/bin/'
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
tf.random.set_seed(1)
from keras import layers
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, GaussianNoise
import time
import matplotlib
import pydot
from keras.utils.vis_utils import plot_model
import itertools
from frechetdist import frdist
import random
import Tools as Tls


def make_generator_model(noise_dim,raman_dim):

    model = tf.keras.Sequential()
    model.add(layers.Input((noise_dim)))
    # Fully Connected Layers
    #(opt) (number of nodes can change and activation may be relu or leaky relu)
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))

    model.add(layers.Dense(256, activation="leaky_relu"))
    model.add(layers.Dense(raman_dim))
    model.compile()

    print(model.output_shape)
    assert model.output_shape == (None, raman_dim)
    return model

def make_discriminator_model(raman_dim):
    # Implementing a ConvNet discriminator
    model = tf.keras.Sequential()

    model.add(layers.Input(shape=(raman_dim)))
    model.add(layers.Reshape([raman_dim, 1]))
    model.add(
        layers.Conv1D(kernel_size=8, filters=256, activation='leaky_relu'))  # (opt) (number of filters and kernel size)
    model.add(layers.MaxPool1D())
    model.add(layers.Dropout(0.2))  # (opt) (dropout probability)

    model.add(layers.Conv1D(kernel_size=8, filters=128))  # (opt) (number of filters and kernel size)
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))

    model.add(layers.MaxPool1D())
    model.add(layers.Dropout(0.2))  # (opt) (dropout probability)

    model.add(layers.Flatten())
    model.add(layers.Dense(64))  # (opt) (number of nodes in layer)
    model.add(layers.Dense(1))
    model.compile()

    return model