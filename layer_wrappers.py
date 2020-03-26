import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import BatchNormalization, Layer, Conv2D

def batch_norm(data_format, momentum=0.997, epsilon=1e-5):
    return BatchNormalization(
            axis=1 if data_format == 'channels_first' else 3,
            momentum=momentum,
            epsilon=epsilon,
            fused=True
        )

def conv2d(filters, kernel_size, strides, data_format):
    return Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same' if strides == 1 else 'valid',
            use_bias=False,
            kernel_initializer=tf.keras.initializers.VarianceScaling(),
            data_format=data_format
        )

def padding(inputs, kernel_size, data_format, **kwargs):
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

class LayerWrapper(Layer):
    """docstring for LayerWrapper"""
    def __init__(self, process_fn, generator_fn, name="LayerWrapper", **kwargs):
        super(LayerWrapper, self).__init__(name=name)
        self.process_fn = process_fn
        self.kwargs = kwargs

        self.layer = generator_fn(**kwargs)

    def call(self, inputs, training=False):
        inputs = self.process_fn(inputs, **self.kwargs)

        return self.layer(inputs)

class Conv(LayerWrapper):
    """docstring for LayerWrapper"""
    def __init__(self, name="Conv", **kwargs):
        super(Conv, self).__init__(process_fn=padding, generator_fn=conv2d, **kwargs)

class BatchNorm(LayerWrapper):
    """docstring for LayerWrapper"""
    def __init__(self, name="BatchNorm", **kwargs):
        super(BatchNorm, self).__init__(process_fn=lambda x, **kwargs: x, generator_fn=batch_norm, **kwargs)
         
if __name__ == "__main__":
    model = Conv(data_format="channels_last", filters=3, kernel_size=3, strides=2)
    test = tf.convert_to_tensor(np.random.normal(size=(1,50,200,3)))
    print(model.call(test))