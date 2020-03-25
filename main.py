import os
import argparse

import numpy as np
import tensorflow as tf

from model import CaptchaModel
from data_frame import DataFrame

def main(args):

    # set memory growth to true to fix potential memory issues
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # get data object
    data = DataFrame(
            path=args.dir, 
            symbols_in_captcha=args.captcha_size, 
            use_lowercase=args.use_lowercase, 
            use_uppercase=args.use_uppercase, 
            use_numbers=args.use_numbers
        )

    # get data split
    (X_train, t_train), (X_test, t_test) = data.get_data(args.test_size)

    # init network
    model = CaptchaModel(
            input_shape=data.get_captcha_dims(), 
            n_linear_blocks=args.captcha_size, 
            n_outputs=data.get_num_symbols()
            # TODO: add more
        )

    # comile network with given params
    model.compile(loss='categorical_crossentropy', optimizer=args.optm, metrics=["accuracy"])

    # train network
    model.fit(X_train, [t_train[i] for i in range(args.captcha_size)], batch_size=32, epochs=50, verbose=1, validation_split=0.05)

    # evaluate performance
    score = model.evaluate(X_test, [t_test[i] for i in range(args.captcha_size)], verbose=1)
    print(score)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dir", type=str, default="./data", help="Directory containing CAPTCHA images")
    parser.add_argument("--captcha_size", type=int, default=5, help="Number of symbols in captcha")
    parser.add_argument("--use_lowercase", type=bool, default=True, help="Indicator if captcha includes lower case symbols")
    parser.add_argument("--use_uppercase", type=bool, default=False, help="Indicator if captcha includes upper case symbols")
    parser.add_argument("--use_numbers", type=bool, default=True, help="Indicator if captcha includes digits")
    parser.add_argument("--optm", type=str, default="adam", help="Optimizer to use")
    parser.add_argument("--test_size", type=float, default=0.1, help="Percent of data to use for test set")

    args = parser.parse_args()
    main(args)
    
