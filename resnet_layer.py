import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Layer, ReLU, MaxPool2D
from tensorflow.keras.models import Sequential

from layer_wrappers import BatchNorm, Conv

def shortcut_fn(filters, strides, data_format):
    """ Shortcut using 1x1 convolution
        
        Args:
            filters: Integer, the dimensionality of the output space (i.e. the
                number of output filters in the convolution).
            strides: An integer or tuple/list of 2 integers, specifying the
                strides of the convolution along the height and width
            data_format: 'channels_first' or 'channels_last'

        Returns:
            A 1x1 convolution layer with padded inputs
    """
    return Conv(
            filters=filters, 
            kernel_size=1, 
            strides=strides, 
            data_format=data_format
        )

class ResNetLayerV2(Layer):
    """ A single ResNetLayer with a skip connection

        Args:
            filters: Integer, the dimensionality of the output space (i.e. the
                number of output filters in the convolution).
            strides: An integer or tuple/list of 2 integers, specifying the
                strides of the convolution along the height and width
            data_format: 'channels_first' or 'channels_last'
            shortcut_fn: callable which returns a desired layer to apply to the
                inputs in the skip connection. If None inputs remain the same
    """
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

        # set shortcut block to be used in __call__()
        self.shortcut = None if shortcut_fn is None else shortcut_fn(filters=filters, strides=strides, data_format=data_format)
        
        # Resnet Layer consists of two consecutive applications
        # of BatchNormalization, ReLU activation and Convolution
        # in that order.
        self.block = Sequential([
                BatchNorm(data_format=data_format),
                ReLU(),
                # first convolution can have strides > 1 effectively downscaling the image
                Conv(filters=filters, kernel_size=3, strides=strides, data_format=data_format),
                BatchNorm(data_format=data_format),
                ReLU(),
                # second convolution always has stires = 1
                Conv(filters=filters, kernel_size=3, strides=1, data_format=data_format)
            ])

    # forward call through layer
    def call(self, inputs, training=False):
        # output of resnet layer
        resnet_output = self.block(inputs, training=training)

        # output from shortcut
        shortcut = inputs if self.shortcut is None else self.shortcut(inputs, training=training)

        # return matrix addition of the two
        return resnet_output + shortcut

# test
if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    test = tf.convert_to_tensor(np.random.normal(size=(3,50,200,3)), dtype=tf.float32)

    model = ResNetLayerV2(filters=3, strides=2, data_format='channels_last', shortcut_fn=shortcut_fn)
    print(model.call(test).shape)

    model = ResNetLayerV2(filters=3, strides=1, data_format='channels_last', shortcut_fn=shortcut_fn)
    print(model.call(test).shape)
