"""
    Messing around with the functional API in keras
    Compare the functional API with the Sequential method
"""

from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras import layers
from tensorflow.python.keras import Input


# Sequential
seq_model = Sequential()
seq_model.add(layers.Dense(32, activation='relu', input_shape=(64, )))
seq_model.add(layers.Dense(32, activation='relu'))
seq_model.add(layers.Dense(10, activation='softmax'))

seq_model.summary()


# Functional
input_tensor = Input(shape=(64, ))
x = layers.Dense(32, activation='relu')(input_tensor)
x = layers.Dense(32, activation='relu')(x)
output_tensor = layers.Dense(10, activation='softmax')(x)

model = Model(input_tensor, output_tensor)

model.summary()



