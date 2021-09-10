from tensorflow.keras import layers
from tensorflow import keras
from keras import backend
import tensorflow as tf
import numpy as np

# Set to True if the used network was generated using progressive growth
DO_GROWTH = False

# Replicate custom layers so the network can be loaded
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
    

class EqualizedDense(layers.Dense):
    def __init__(self, *args, gain=1, **kwargs):
        self.scale = 1.0
        self.gain = gain
        super(EqualizedDense, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        fan_in = np.product([int(val) for val in input_shape[1:]])
        self.scale = self.gain / np.sqrt(fan_in)
        return super(EqualizedDense, self).build(input_shape)

    def call(self, inputs):
        outputs = backend.dot(inputs, self.kernel*self.scale)
        if not DO_GROWTH:
            outputs = backend.dot(inputs, self.kernel)
        if self.use_bias:
            outputs = backend.bias_add(outputs, self.bias, data_format='channels_last')
        if self.activation is not None:
            return self.activation(outputs)
        return outputs


def genHeightMap():
    # Number of images to generate
    num_img = 1
    # Latent noise dimension of generator
    noise_dim = 512

    generator = keras.models.load_model("Generator.h5", custom_objects={
        "EqualizedConv2D": EqualizedConv2D,
        "EqualizedDense": EqualizedDense
    })
    random_latent_vectors = tf.random.normal(shape=(num_img, noise_dim))

    # Generate output of network
    generated_images = generator.predict(random_latent_vectors)
    # Scale output to allow conversion to image
    generated_images = (generated_images * 127.5) + 127.5
    img = generated_images[0]
    
    # Convert array to image
    img = keras.preprocessing.image.array_to_img(img)
    return img
