import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Layer, ReLU, MaxPool2D
from tensorflow.keras.models import Sequential

from layer_wrappers import BatchNorm, Conv

def shortcut_fn(filters, strides, data_format, kernel_size=1):
    return Conv(
            filters=filters, 
            kernel_size=kernel_size, 
            strides=strides, 
            data_format=data_format
        )

class ResNetLayerV2(Layer):

    def __init__(
                self, 
                filters, 
                strides, 
                data_format, 
                shortcut_fn=None, 
                name='ResNetLayerV2', 
                **kwargs
            ):
        super(ResNetLayerV2, self).__init__(name=name, **kwargs)

        self.shortcut = None if shortcut_fn is None else shortcut_fn(filters=filters, strides=strides, data_format=data_format)
         
        self.block = Sequential([
                BatchNorm(data_format=data_format),
                ReLU(),
                Conv(filters=filters, kernel_size=3, strides=strides, data_format=data_format),
                BatchNorm(data_format=data_format),
                ReLU(),
                Conv(filters=filters, kernel_size=3, strides=1, data_format=data_format)
            ])

    def call(self, inputs, training=False):
        output = self.block(inputs, training=training)
        shortcut = inputs if self.shortcut is None else self.shortcut(inputs, training=training)

        print(output)
        print(shortcut)

        return output + shortcut

if __name__ == "__main__":
    model = ResNetLayerV2(filters=3, strides=2, data_format='channels_last', shortcut_fn=shortcut_fn)
    test = tf.convert_to_tensor(np.random.normal(size=(3,50,200,3)), dtype=tf.float32)
    print(model.call(test))