import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import MaxPool2D, ReLU, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential, Model

from layer_wrappers import BatchNorm, Conv
from resnet_block import ResNetBlock

class ResNet(Model):
    """docstring for ResNet"""
    def __init__(
                self, 
                n_classes, 
                n_filters, 
                kernel_size,
                first_conv_stride,
                first_pool_size,
                first_pool_stride,
                block_sizes,
                block_strides,
                data_format,
                name="ResNet", 
                **kwargs
            ):
        super(ResNet, self).__init__(name=name, **kwargs)

        self.network = Sequential([
                Conv(filters=n_filters, kernel_size=kernel_size, strides=first_conv_stride, data_format=data_format)    
            ])

        if first_pool_size:
            self.network.add(MaxPool2D(pool_size=first_pool_size, strides=first_pool_stride, padding='same', data_format=data_format))

        for idx, n_blocks in enumerate(block_sizes):
            filters = n_filters * (2**idx)

            self.network.add(ResNetBlock(n_layers=n_blocks, filters=filters, strides=block_strides[idx], data_format=data_format, name=f"ResNetBlock({idx})"))

        self.network.add(BatchNorm(data_format=data_format))
        self.network.add(ReLU())
        self.network.add(AveragePooling2D(pool_size=2, strides=2, padding='same', data_format=data_format))
        self.network.add(Flatten(data_format=data_format))
        self.network.add(Dense(units=n_classes))


    def call(self, inputs, training=False):
        return self.network(inputs, training=training)

if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    test = tf.convert_to_tensor(np.random.normal(size=(3,50,200,1)), dtype=tf.float32)

    model = ResNet(
                    n_classes=10,
                    n_filters=64, 
                    kernel_size=7,
                    first_conv_stride=2,
                    first_pool_size=3,
                    first_pool_stride=2,
                    block_sizes=[2, 2, 2, 2],
                    block_strides=[1, 2, 2, 2],
                    data_format='channels_last',
                )

    print(model(test).shape)
    
        