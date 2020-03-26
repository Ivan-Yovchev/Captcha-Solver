import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Layer, ReLU, MaxPool2D
from tensorflow.keras.models import Sequential

from layer_wrappers import BatchNorm, Conv

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

        self.shortcut_fn = shortcut_fn
        
        # a single block consists 
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
        shortcut = inputs if self.shortcut_fn is None else self.shortcut_fn(inputs, training)

        print(output)
        print(shortcut)

        return output + shortcut

if __name__ == "__main__":
    model = ResNetLayerV2(filters=64, strides=2, data_format='channels_last')
    test = tf.convert_to_tensor(np.random.normal(size=(3,50,200,3)), dtype=tf.float32)
    # print(test)
    test = Conv(filters=64, kernel_size=7, strides=2, data_format='channels_last')(test)
    test = MaxPool2D(pool_size=3, strides=2, data_format='channels_last')(test)
    print(model.call(test))
    # print(test)