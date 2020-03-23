import os
import argparse

import numpy as np

def main(args):
    pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dir", type=str, default="./data", help="Directory containing CAPTCHA images")

    args = parser.parse_args()
    main(args)
    
