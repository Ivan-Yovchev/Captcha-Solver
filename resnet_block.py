import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential

from resnet_layer import ResNetLayerV2, shortcut_fn

class ResNetBlock(Layer):
    """docstring for ResNetBlock"""
    def __init__(
                self, 
                n_layers, 
                filters, 
                strides, 
                data_format,
                shortcut=shortcut_fn,
                name="ResNetBlock", 
                **kwargs
            ):
        super(ResNetBlock, self).__init__(name=name, **kwargs)

        self.block = Sequential()

        for idx in range(n_layers):
            self.block.add(ResNetLayerV2(filters=filters, strides=strides, data_format=data_format, shortcut_fn=shortcut))

            if idx == 0:
                strides = 1
                shortcut = None

    def call(self, inputs, training=False):
        return self.block(inputs, training=training)

if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    test = tf.convert_to_tensor(np.random.normal(size=(3,50,200,3)), dtype=tf.float32)

    model = ResNetBlock(n_layers=2, filters=3, strides=1, data_format='channels_last')
    print(model(test).shape)

    model = ResNetBlock(n_layers=2, filters=3, strides=2, data_format='channels_last')
    print(model(test).shape)