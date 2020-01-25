#!/usr/bin/env python3.7
import normaliser
import sys

# parse arguments from command line:
test_file = sys.argv[1]
train_file = sys.argv[2]
norm_type = sys.argv[3]

if __name__ == "__main__":
    normaliser.transform_test_data(test_file, train_file, norm_type)

