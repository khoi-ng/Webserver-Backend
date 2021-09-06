from tensorflow.keras import layers
from tensorflow import keras
from keras import backend
import tensorflow as tf
import numpy as np


DO_GROWTH = False


class EqualizedConv2D(layers.Conv2D):
    def __init__(self, *args, **kwargs):
        self.scale = 1.0
        super(EqualizedConv2D, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        fan_in = np.product([int(val) for val in input_shape[1:]])
        self.scale = np.sqrt(2/fan_in)
        return super(EqualizedConv2D, self).build(input_shape)

    def call(self, inputs):
        outputs = backend.conv2d(inputs, self.kernel * self.scale, strides=self.strides, padding=self.padding,
                                 data_format=self.data_format, dilation_rate=self.dilation_rate)
        if not DO_GROWTH:
            outputs = backend.conv2d(inputs, self.kernel, strides=self.strides, padding=self.padding, data_format=self.data_format, dilation_rate=self.dilation_rate)
        if self.use_bias:
            outputs = backend.bias_add(outputs, self.bias, data_format=self.data_format)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs
    


def genHeightMap():
    num_img = 1
    noise_dim = 512

    generator = keras.models.load_model("Generator.h5", custom_objects={
        "EqualizedConv2D": EqualizedConv2D
    })
    random_latent_vectors = tf.random.normal(shape=(num_img, noise_dim))

    generated_images = generator.predict(random_latent_vectors)
    generated_images = (generated_images * 127.5) + 127.5
    img = generated_images[0]
    
    img = keras.preprocessing.image.array_to_img(img)
    # img.save("output\\generated_img_{i}.png".format(i=1))
    return img
