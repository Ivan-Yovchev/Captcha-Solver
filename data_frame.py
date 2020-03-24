import os
import cv2
import string

import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split

class DataFrame(object):
    """docstring for DataFrame"""
    def __init__(self, path, symbols_in_captcha, use_lowercase=False, use_uppercase=False, use_numbers=False):
        super(DataFrame, self).__init__()

        # store path
        self.path = path
        # number of symbols iin captcha
        self.symbols_in_captcha = symbols_in_captcha

        # init all possible symbols
        self.symbols = ''

        # add to symbols according to params
        if use_lowercase:
            self.symbols += string.ascii_lowercase
        if use_uppercase:
            self.symbols += string.ascii_uppercase
        if use_numbers:
            self.symbols += '0123456789'

        assert(len(self.symbols) != 0)

        # number of all possible symbols
        self.n_symbols = len(self.symbols)

        # get all files in dir
        files = os.listdir(self.path)

        # store number of images in dir
        self.n_captchas = len(files)

        # get first image to extract captcha sizes
        self.captcha_dims = cv2.imread(os.path.join(path, files[0]), cv2.IMREAD_GRAYSCALE).shape + (1, )

        # initialize storage for data inputs and targets
        self.X = np.zeros((self.n_captchas,) + self.captcha_dims)
        self.t = np.zeros((self.symbols_in_captcha, self.n_captchas, self.n_symbols))

        for i, captcha_file in enumerate(files):

            # read image as gray scale and normalize
            img = cv2.imread(os.path.join(path, captcha_file), cv2.IMREAD_GRAYSCALE) / 255.

            # unsquuze last dim due to gray scale
            # (x, y) -> (x, y, 1) equivalent to single channel image
            img = np.expand_dims(img, axis=len(img.shape))

            # file name is target not including file extension
            target = captcha_file.split(".")[0]

            assert(len(target) == self.symbols_in_captcha)

            # convert target to indecies
            target = self.string_to_indecies(target, self.symbols)

            # convert to one hots
            target = tf.one_hot(target, self.n_symbols)

            # store input and target values
            self.X[i] = img
            self.t[:, i] = target.numpy()

    def string_to_indecies(self, chars, full_str):

        # array to keep track of idxs
        indecies = []
        for ch in chars:
            # idx of given char
            idx = full_str.find(ch)
            
            # assert not -1 aka not found
            assert(idx > -1)

            # add to array
            indecies.append(idx)

        return indecies

    def get_data(self, test_size=0.2):
        pass

if __name__ == "__main__":
    test = DataFrame("./data", 5, use_lowercase=True, use_numbers=True)