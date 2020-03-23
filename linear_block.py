import tensorflow as tf

from tensorflow.keras.layers import Layer, Dense, Dropout
from tensorflow.keras.models import Sequential

class LinearBlock(Layer):

    def __init__(self, n_output, n_hidden=64, dropout=0.5, hidden_activation='relu', name='LinearBlock', **kwargs):
        super(LinearBlock, self).__init__(name=name, **kwargs)
        
        self.block = Sequential([
                Dense(units=n_hidden, activation=hidden_activation),
                Dropout(rate=dropout),
                Dense(units=n_output, activation='sigmoid')
            ])

    def call(self, inputs):
        return self.block(inputs)

if __name__ == "__main__":

    # test
    x = tf.random.normal((4, 128,))
    block = LinearBlock(5)
    y = block(x)
    print(y)