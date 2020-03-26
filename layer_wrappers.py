import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import BatchNormalization, Layer, Conv2D

###############################################################################
#                       LAYER CONSTRUCTOR FUNCTIONS                           #
###############################################################################

def batch_norm(data_format, momentum=0.997, epsilon=1e-5):
    """ BatchNormalization using a standard set of params

    Args:

        data_format: 'channels_first' or 'channels_last'
        momentum: momentum for the moving average
        epsilon: small float added to variance to avoid dividing by zero

    Returns:
        A BatchNormalization tensorflow layer object
    """

    return BatchNormalization(
            axis=1 if data_format == 'channels_first' else 3,
            momentum=momentum,
            epsilon=epsilon,
            fused=True
        )

def conv2d(filters, kernel_size, strides, data_format):
    """ BatchNormalization using a standard set of params

    Args:
        filters: Integer, the dimensionality of the output space (i.e. the
            number of output filters in the convolution).
        kernel_size: an integer or tuple/list of 2 integers, specifying the
            height and width of the 2D convolution window
        strides: An integer or tuple/list of 2 integers, specifying the
            strides of the convolution along the height and width
        data_format: 'channels_first' or 'channels_last'

    Returns:
        A Conv2D tensorflow layer object
    """

    return Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=('same' if strides == 1 else 'valid'),
            use_bias=False,
            kernel_initializer=tf.keras.initializers.VarianceScaling(),
            data_format=data_format
        )

###############################################################################
#                        INPUT PROCESSING FUNCTIONS                           #
###############################################################################

def padding(inputs, kernel_size, data_format, **kwargs):
    """ Pad input with zeros 

    Args:
        inputs: A tensor of a shape given by 'data_format'
        kernel_size: The kernel used by conv or maxpool
        data_format: 'channels_first' or 'channels_last').

    Returns:
        Unchaged input tensor if kernel_size is 1 or padded (kernel_size > 1).
    """

    total_pad = kernel_size - 1
    padding_start = total_pad // 2
    padding_end = total_pad - padding_start

    paddings = None
    if data_format == 'channels_first':
        paddings = [[0,0], [0,0], [padding_start, padding_end], [padding_start, padding_end]]
    elif data_format == 'channels_last':
        paddings = [[0,0], [padding_start, padding_end], [padding_start, padding_end], [0,0]]

    assert(paddings is not None)

    return tf.pad(tensor=inputs, paddings=paddings)

###############################################################################
#                        WRAPPER CLASS DEFINITIONS                            #
###############################################################################

class LayerWrapper(Layer):
    """ Parent Wrapper Class

    Args:
        process_fn: callable to preprocess input tensor before the forward
            __call__() through the layer
        generator_fn: callable which returns a tensorflow layer of the desired
            type

    """
    def __init__(self, process_fn, generator_fn, name="LayerWrapper", **kwargs):
        super(LayerWrapper, self).__init__(name=name)

        # store process_fn to be used in the forward call
        self.process_fn = process_fn

        # save kwargs for the process function in the forward call
        self.kwargs = kwargs

        # save layer object for forward pass
        self.layer = generator_fn(**kwargs)

    def call(self, inputs, training=False):

        # preprocess inputs
        inputs = self.process_fn(inputs, **self.kwargs)

        # return results from forward pass through layer
        return self.layer(inputs, training=training)

class Conv(LayerWrapper):
    """ Child for LayerWrapper which uses a Conv2D layer """
    def __init__(self, name="Conv", **kwargs):
        super(Conv, self).__init__(process_fn=padding, generator_fn=conv2d, **kwargs)

class BatchNorm(LayerWrapper):
    """ Child for LayerWrapper which uses a BatchNormalization layer """
    def __init__(self, name="BatchNorm", **kwargs):
        # no preprocesing for BatchNorm
        # we can simply use a lambda to return the unchaged inputs
        super(BatchNorm, self).__init__(process_fn=lambda x, **kwargs: x, generator_fn=batch_norm, **kwargs)

# test
if __name__ == "__main__":
    model = Conv(data_format="channels_last", filters=3, kernel_size=3, strides=2)
    test = tf.convert_to_tensor(np.random.normal(size=(1,50,200,3)))
    print(model.call(test))