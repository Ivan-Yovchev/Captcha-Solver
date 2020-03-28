import os
import argparse

import numpy as np
import tensorflow as tf

from data_frame import DataFrame

from resnet import ResNet18


def main(args):

    # set memory growth to true to fix potential memory issues
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # get data object
    data = DataFrame(
            path=args.dir, 
            n_symbols_in_captcha=args.captcha_size, 
            use_lowercase=args.use_lowercase, 
            use_uppercase=args.use_uppercase, 
            use_numbers=args.use_numbers
        )

    # get data split
    (X_train, t_train), (X_test, t_test) = data.get_data(args.test_size)

    # init network
    model = ResNet18(n_classes=(data.get_num_symbols() * args.captcha_size), data_format='channels_last')

    # comile network with given params
    model.compile(loss='binary_crossentropy', optimizer=args.optm, metrics=["accuracy"])

    # train network
    model.fit(X_train, t_train, batch_size=args.batch_size, epochs=args.epochs, verbose=1)

    # evaluate performance
    score = model.evaluate(X_test, t_test, verbose=1)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dir", type=str, default="./data", help="Directory containing CAPTCHA images")
    parser.add_argument("--captcha_size", type=int, default=5, help="Number of symbols in captcha")
    parser.add_argument("--use_lowercase", type=bool, default=True, help="Indicator if captcha includes lower case symbols")
    parser.add_argument("--use_uppercase", type=bool, default=False, help="Indicator if captcha includes upper case symbols")
    parser.add_argument("--use_numbers", type=bool, default=True, help="Indicator if captcha includes digits")
    parser.add_argument("--optm", type=str, default="adam", help="Optimizer to use")
    parser.add_argument("--test_size", type=float, default=0.2, help="Percent of data to use for test set")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs to train on")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")

    args = parser.parse_args()
    main(args)
    
