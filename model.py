import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten
from tensorflow.keras.models import Sequential

from linear_block import LinearBlock

class CaptchaModel(tf.keras.models.Model):
    """docstring for Model"""
    def __init__(
            self, 
            input_shape, 
            n_linear_blocks,
            n_outputs,
            convs=[16, 32, 32], 
            linear_n_hidden=64, 
            linear_dropout=0.5, 
            linear_h_activation='relu', 
            name="captcha_model", 
            **kwargs):
        super(CaptchaModel, self).__init__()

        self.n_linear_blocks = n_linear_blocks

        self.conv = Sequential([
                Input(shape=input_shape),
                Conv2D(convs[0], 3, padding='same', activation='relu'),
                MaxPooling2D(padding='same'),
                Conv2D(convs[1], 3, padding='same', activation='relu'),
                MaxPooling2D(padding='same'),
                Conv2D(convs[2], 3, padding='same', activation='relu'),
                BatchNormalization(),
                MaxPooling2D(),
                Flatten()
            ])

        self.linear_blocks = []
        for _ in range(n_linear_blocks):
            block = LinearBlock(n_outputs, linear_n_hidden, linear_dropout, linear_h_activation)
            self.linear_blocks.append(block)

    def call(self,  inputs):
        conv = self.conv(inputs)
        print("Conv: ", conv.shape)
        result = []
        for i in range(self.n_linear_blocks):
            letter_result = self.linear_blocks[i](conv)
            result.append(letter_result)

        return result

if __name__ == "__main__":

    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    x = tf.random.normal((2, 28, 28, 1))
    model = CaptchaModel(x.shape[1:], 5, 5)
    y = model(x)
    print(y.)
        