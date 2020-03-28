import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import MaxPool2D, ReLU, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential, Model

from layer_wrappers import BatchNorm, Conv
from resnet_block import ResNetBlock

class ResNet(Model):
    """ ResNet V2 architecture

        Args:
            n_classes: Integer, number of classes for the output dimension
                of the network
            first_conv_n_filters: Integer, number of filters to be used for the 
                first conv layer. For each subsiquent conv layer the number of 
                filters is doubled
            first_conv_kernel_size: Integer, kernel size to be used for the 
                first conv layer
            first_conv_stride: Integer, the stride size to be used for the
                first conv layer
            block_sizes: List of integers, the number of ResNetLayers to be
                contained in a ResNetBlock
            block_strides: List of integers, must have the same size as 
                block_sizes. Each value represents the stride size to be used
                for the first conv of a ResNetLayer
            data_format: 'channels_first' or 'channels_last'
            first_pool_size: Integer, the pool kernel size to be used after the
                first convolution. If None MaxPooling is not applied after the
                first conv.
            first_pool_stride: Integer, stride size for pooling layer

        Raises:
            ValueError: if length of block_sizes and block_strides does not 
                match
    """
    def __init__(
                self, 
                n_classes, 
                first_conv_n_filters, 
                first_conv_kernel_size,
                first_conv_stride,
                block_sizes,
                block_strides,
                data_format,
                first_pool_size=None,
                first_pool_stride=None,
                name="ResNet", 
                **kwargs
            ):
        super(ResNet, self).__init__(name=name, **kwargs)

        # TODO: Add channels switch optimization

        # perform check to make sure list sizes are the same
        if len(block_sizes) != len(block_strides):
            raise ValueError('Lists block_sizes and block_strides must be of the same size')

        # initialize network with a single conv for now
        self.network = Sequential([
                Conv(filters=first_conv_n_filters, kernel_size=first_conv_kernel_size, strides=first_conv_stride, data_format=data_format)    
            ])

        # if first_pool_size is not None add a pooling layer to the network
        if first_pool_size:
            self.network.add(MaxPool2D(pool_size=first_pool_size, strides=first_pool_stride, padding='same', data_format=data_format))

        # the number of ResNetBlocks is based on the size of the two lists
        # block_sizes and block_strides
        for idx, n_blocks in enumerate(block_sizes):

            # double the number of filters for the next layer
            filters = first_conv_n_filters * (2**idx)

            # add a ResNetBlock to the network
            self.network.add(ResNetBlock(n_layers=n_blocks, filters=filters, strides=block_strides[idx], data_format=data_format))

        # once all the ResNetBlocks are added
        # add final BatchNorm and activation layers
        self.network.add(BatchNorm(data_format=data_format))
        self.network.add(ReLU())

        # add aglobal average pooling layer
        # to go from B x H x W x C (or B x C x H x W)
        # to B x C
        self.network.add(GlobalAveragePooling2D(data_format=data_format))

        # add Dense layer to perform final classification
        self.network.add(Dense(units=n_classes, activation='sigmoid'))

    def call(self, inputs, training=False):
        # forward call through network
        return self.network(inputs, training=training)

class ResNet18(ResNet):
    """ ResNet Wrapper with the ResNet18 default params"""
    def __init__(self, n_classes, data_format, name="ResNet18", **kwargs):
        super(ResNet18, self).__init__(
                                        n_classes=n_classes,
                                        first_conv_n_filters=64, 
                                        first_conv_kernel_size=7,
                                        first_conv_stride=2,
                                        first_pool_size=3,
                                        first_pool_stride=2,
                                        block_sizes=[2, 2, 2, 2],
                                        block_strides=[1, 2, 2, 2],
                                        data_format=data_format,
                                        name=name, 
                                        **kwargs
                                    )

# TODO: Add ResNet34, ResNet50, ResNet101, ResNet152

# test
if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    test = tf.convert_to_tensor(np.random.normal(size=(3,50,200,1)), dtype=tf.float32)

    model = ResNet18(n_classes=10, data_format='channels_last')

    print(model(test).shape)
    
        