"""
    Visualizing convnet filter
"""

import os
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    x += 0.5
    x = np.clip(x, 0, 1)

    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def generate_pattern(layer_name, filter_index, size=150):
    # Defining loss tensor for filter viz
    layer_output = mdl.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # Obtaining grad of loss w.r.t. input
    grads = K.gradients(loss, mdl.input)[0]

    # Grad-normalization trick
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([mdl.input], [loss, grads])

    # Loss maximization via SGD
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.

    step = 1
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value*step

    img = input_img_data[0]
    return deprocess_image(img)



mdl = VGG16(weights='imagenet', include_top=False)

layers_list = ['block1_conv1',
               'block2_conv1',
               'block3_conv1',
               'block4_conv1',
               'block5_conv1']

for l in layers_list:
    plt.imshow(generate_pattern(l, 0))
    plt.title(l)
    plt.show()





