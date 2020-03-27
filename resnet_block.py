import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential

from resnet_layer import ResNetLayerV2, shortcut_fn


class ResNetBlock(Layer):
    """ A ResNetBlock consisting of mutlitple ResNetLayers

        Args:
            n_filters: Integer, the number of ResNetLayers contained in the
                block
            filters: Integer, the dimensionality of the output space (i.e. the
                number of output filters in the convolution).
            strides: An integer or tuple/list of 2 integers, specifying the
                strides of the convolution along the height and width
            data_format: 'channels_first' or 'channels_last'
            shortcut: callable which returns a desired layer to apply to the
                inputs in the skip connection. If None inputs remain the same.
                Dafaults to shortcut_fn from resnet_layer
    """
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

        # initialize ResNetBlock
        self.block = Sequential()

        # add n_layers number of ResNetLayers
        for idx in range(n_layers):
            self.block.add(ResNetLayerV2(filters=filters, strides=strides, data_format=data_format, shortcut_fn=shortcut))

            # only first ResNetLayer uses strides and a shortcut
            # any subsiquent ResNetLayers just use strides = 1
            # and no shortcut
            if idx == 0:
                strides = 1
                shortcut = None

    def call(self, inputs, training=False):
        # forward pass through block
        return self.block(inputs, training=training)

# test
if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    test = tf.convert_to_tensor(np.random.normal(size=(3,50,200,3)), dtype=tf.float32)

    model = ResNetBlock(n_layers=2, filters=3, strides=1, data_format='channels_last')
    print(model(test).shape)

    model = ResNetBlock(n_layers=2, filters=3, strides=2, data_format='channels_last')
    print(model(test).shape)