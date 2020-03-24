import os
import argparse

import numpy as np

from model import CaptchaModel
from data_frame import DataFrame

def main(args):

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

    model.summary()

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
    
