import tensorflow as tf

from tensorflow.keras.layers import Layer, Dense, Dropout
from tensorflow.keras.models import Sequential

class LinearBlock(Layer):

    def __init__(self, n_output, n_hidden=64, dropout=0.5, hidden_activation='relu', name='LinearBlock', **kwargs):
        super(LinearBlock, self).__init__(name=name, **kwargs)
        
        # a single block consists 
        self.block = Sequential([
                # of a linear layer from inputs to hidden
                Dense(units=n_hidden, activation=hidden_activation),
                # dropout against overfitting
                Dropout(rate=dropout),
                # and linear to outputs
                Dense(units=n_output, activation='sigmoid')
            ])

    def call(self, inputs):
        return self.block(inputs)