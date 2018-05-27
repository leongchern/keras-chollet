"""
    Messing around with the functional API in keras
"""

from tensorflow.python.keras import Input, layers
from tensorflow.python.keras.models import Model
import numpy as np

x_train = np.random.random((1000,64))
y_train = np.random.random((1000,10))

input_tensor = Input((64, ))
x = layers.Dense(32, activation='relu')(input_tensor)
x = layers.Dense(32, activation='relu')(x)
output_tensor = layers.Dense(10, activation='softmax')(x)

model = Model(input_tensor, output_tensor)

# model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=8)

score = model.evaluate(x_train, y_train)




