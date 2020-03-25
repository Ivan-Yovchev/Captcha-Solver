import math

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

        # number of linear blocks is dynmic so store value for later
        self.n_linear_blocks = n_linear_blocks

        # Basic convolution
        self.conv = Sequential([
                # init input shape
                Input(shape=input_shape),
                # 2d conv with 3 filters
                Conv2D(convs[0], 3, padding='same', activation='relu'),
                # max pooling to reduce size
                MaxPooling2D(padding='same'),
                # another 2d with 3 filters
                Conv2D(convs[1], 3, padding='same', activation='relu'),
                # reduce size again
                MaxPooling2D(padding='same'),
                # last conv with 3 filters
                Conv2D(convs[2], 3, padding='same', activation='relu'),
                # batch norm for faster learning
                BatchNormalization(),
                # reduce size
                MaxPooling2D(),
                # turn to vector
                Flatten()
            ])

        # captches consist of n number of letters/numbers
        # create a subnet for each one dynamically
        self.linear_blocks = []
        for _ in range(n_linear_blocks):
            # create block
            block = LinearBlock(n_outputs, linear_n_hidden, linear_dropout, linear_h_activation)

            # add to array to use in the forward pass later
            self.linear_blocks.append(block)

    # forward pass
    def call(self,  inputs):

        # get conv flattened vector
        conv = self.conv(inputs)

        # to store results from each subnet
        result = []

        # for each subnet
        for i in range(self.n_linear_blocks):
            # forwrd pass conv vector through net
            letter_result = self.linear_blocks[i](conv)

            # store results
            result.append(letter_result)

        return result

    def custom_evaluate(self, X, t, batch_size):

        # n_captcha x samples x 1
        t = tf.argmax(t, axis=-1)

        # counter for number of correct classifications
        correct = 0

        # number of samples in test set
        n_samples = X.shape[0]

        # number of batches
        n_batches = math.ceil(n_samples / batch_size)

        # for each batch
        for batch in range(n_batches):

            # calcuate end points of batch
            start = batch * batch_size
            end = (batch+1) * batch_size

            # forward pass through network with batch
            results = self(X[start:end,:])

            batch_correct = None

            # for each letter in the captcha
            for i in range(len(results)):

                # compare with targets
                compare_matrix = tf.equal(tf.argmax(results[i], axis=-1), t[i,start:end])

                # first time only
                if batch_correct == None:
                    batch_correct = compare_matrix
                    continue

                # logical and with letters so far
                batch_correct = tf.logical_and(batch_correct, compare_matrix)

            # add number of correct classifications
            correct += tf.reduce_sum(tf.cast(batch_correct, tf.int32)).numpy()

        return correct, X.shape[0], correct / X.shape[0]
        