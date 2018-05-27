"""
    Cat dog classification with data augmentation
"""

import os
from tensorflow.python.keras.preprocessing import image
import matplotlib.pyplot as plt


# Get directories
base_dir = 'catdog_img'

train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir,'val')
test_dir = os.path.join(base_dir,'test')

train_cats_dir = os.path.join(train_dir, 'cats')
val_cats_dir = os.path.join(val_dir, 'cats')
test_cats_dir = os.path.join(test_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
val_dogs_dir = os.path.join(val_dir, 'dogs')
test_dogs_dir = os.path.join(test_dir, 'dogs')


# Setting up the data aug config
datagen = image.ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


fnames = [os.path.join(train_cats_dir, fn) for fn in os.listdir(train_cats_dir)]
img_path = fnames[3]
img = image.load_img(img_path, target_size=(150, 150))
x = image.img_to_array(img)
x = x.reshape((1, ) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break

plt.show()