"""
    Visualizing intermediate activations
"""

# from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import models
from tensorflow.python.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

mdl = models.load_model('catdog_mdl\cats_and_dogs_small_2.h5')
mdl.summary()


# Pre-processing a single image
img_path = 'catdog_img/test/cats/cat.1700.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255
print('shape of     image tensor', img_tensor.shape)
# plt.imshow(img_tensor[0])
# plt.show()


# Instantiating a model **
layer_outputs = [layer.output for layer in mdl.layers[:8]]
activation_model = models.Model(inputs=mdl.input, outputs=layer_outputs)


# Running the model in predict mode
activations = activation_model.predict(img_tensor)
first_layer_act = activations[0]
print('shape of first layer activation', first_layer_act.shape)


# Visualizing 4th channel
plt.matshow(first_layer_act[0, :, :, 4], cmap='viridis')
plt.title('Fourth channel of image')
plt.show()

plt.matshow(first_layer_act[0, :, :, 7], cmap='viridis')
plt.title('Seventh channel of image')
plt.show()


# Visualizing every channel in every intermediate activation
layer_names_list = []
for l in mdl.layers[:8]:
    layer_names_list.append(l.name)

images_per_row = 16

for layer_name, layer_act in zip(layer_names_list, activations):
    n_features = layer_act.shape[-1]
    size = layer_act.shape[1]
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size*n_cols, images_per_row*size))

    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_act[0, :, :, col*images_per_row+row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col*size : (col+1)*size, row*size : (row+1)*size] = channel_image

        scale = 1./size
        plt.figure(figsize=(scale*display_grid.shape[1],
                            scale*display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()



