from tensorflow.keras.layers import MaxPool2D, ReLU, AveragePooling2D, Flatten
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
    pass
        