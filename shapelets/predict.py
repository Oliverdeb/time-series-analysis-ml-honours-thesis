import argparse
from keras.models import load_model
import numpy as np

def predict(load_file=None):
    if args.load:
        model = load_model(load_file)
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        test_name = 'test_data_output' + load_file
        from pickle import load
        test = load(open(test_name))
        print (test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-load", help="load model from file", default=None)


    args = parser.parse_args()

    predict(args.load)

    
