import os
from tensorflow.python.keras.datasets import mnist

from tensorflow.python.keras import models
from tensorflow.python.keras import layers

from tensorflow.python.keras.utils import to_categorical


os.environ['TF_CPP_MIN_LONG_LEVEL'] = '2'


# Loading data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# print(train_images.shape)
# print(len(train_labels))
# print(test_labels)

# Architecture
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

# Compilation step
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Preparing image data
train_images = train_images.reshape((60000,  28 * 28))    # (60000, 28, 28) to (60000, 784)
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# Preparing labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Fitting model
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# Evaluate model
test_loss, test_acc = network.evaluate(test_images, test_labels)
print(test_loss, test_acc)
