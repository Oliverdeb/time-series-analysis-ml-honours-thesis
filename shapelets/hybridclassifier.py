import tensorflow as tf
from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.utils import multi_gpu_model
from keras.optimizers import SGD
from scipy.stats import zscore
import pandas as pd
import numpy as np
import argparse



class LSTMClassifier:

    def __init__(self, input_file):
        self.input_file = input_file

    def load_data(self, test_split = 0.33):
        print ('Loading data...')
        df = pd.read_csv(self.input_file)

        # convert to float
        df['sequence'] = df['sequence'].apply(lambda x: [float(e) for e in x.split()[:20]])
        # standardize each shapelet individually
        df['sequence'] = df['sequence'].apply(zscore)
        df = df[df.sequence.map(lambda x: False in np.isnan(x))]

        # store number of classes in dataset
        self.n_classes = df['target'].values[-1] + 1

        # randomize ordering
        df = df.reindex(np.random.permutation(df.index))

        train_size = int(len(df) * (1 - test_split))
        # split into training and test sets
        X_train = np.array(df['sequence'].values[:train_size])
        y_train = np.array(df['target'].values[:train_size])
        X_test = np.array(df['sequence'].values[train_size:])
        y_test = np.array(df['target'].values[train_size:])
        
        # pad sequences
        X_train = pad_sequences(X_train, dtype='float64')
        X_test = pad_sequences(X_test, dtype='float64')

        return \
            X_train.reshape(len(X_train), len(X_train[0]), 1), \
            y_train, \
            X_test.reshape(len(X_test), len(X_test[0]), 1), \
            y_test


    def create_model(self, n):
        print ('Creating model...')
        with tf.device('/cpu:0'):
            model = Sequential()

            model.add(LSTM(
                128,
                recurrent_activation="hard_sigmoid",
                activation="sigmoid",
                input_shape=(20, 1),
                return_sequences=True))
            model.add(Dropout(0.25))

            model.add(LSTM(
                128,
                recurrent_activation="hard_sigmoid",
                activation="sigmoid"))
            model.add(Dropout(0.5))

            model.add(Dense(self.n_classes, activation='softmax'))
        print ('Compiling...')
        opt = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        opt = 'adam'
        if n != 0:
            parallel_model = multi_gpu_model(model, gpus=n)
            parallel_model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
            return model, parallel_model
        else:
            model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
            return model

        # self.model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # def train(self, batch_size, epochs):
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-load", help="load model from file", default=None)
    parser.add_argument("-gpu", help="number of gpus to run on", required=True)

    args = parser.parse_args()

    file_name = 'new_out400-20-30-30-37.csv'
    file_name = 'std_186892-20-20-0.7-1256.csv'
    input_file = 'output/' + file_name

    classifier = LSTMClassifier(input_file)

    if args.load is None:
        X_train, y_train, X_test, y_test = classifier.load_data()
        # exit(1)
        n_gpus = int(args.gpu)
        print ('Fitting model...')
        if n_gpus != 0:
            original_model, parallel_model = classifier.create_model(n_gpus)
            model = parallel_model
        else:
            original_model = classifier.create_model(n_gpus)
            model = original_model
            n_gpus = 1
        batch_size = 128
        epochs = 600
        
        # classifier.train(batch_size, epochs)
        hist = model.fit(X_train, y_train, batch_size=batch_size*n_gpus, epochs=epochs, validation_split = 0.1, verbose = 1)

        model.summary()
        original_model.save("mse-rmsprop-{0}_classes_128_batchsize={1}_epochs_{2}_model-{3}.h5".format (classifier.n_classes, batch_size, epochs, file_name.replace('.csv', '')))
        
        score, acc = model.evaluate(X_test, y_test, batch_size=128)

        print('Test score:', score)
        print('Test accuracy:', acc)
    else:
        print ("Loading model from file...")
        model = load_model(args.load)        
    pls = []
    # for line in open('../data/snp2.csv').readlines()[100:110]:
        # pls.append([float(x) for x in line.split(',')[1].split(' ')][:20])
    
    test = "1978.09 1953.03 1961.05 1952.29 1942.04 1969.41 1921.22 1951.13 1948.86 1913.85 1972.18 1988.87 1987.66 1940.51 1867.61 1893.21 1970.89 2035.73 2079.61 2096.92"
    test2 = "2270.75 2257.83 2238.83 2249.26 2249.92 2268.88 2263.79 2260.96 2265.18 2270.76 2262.53 2258.07 2262.03 2253.28 2271.72 2256.96 2259.53 2246.19 2241.35 2212.23"
    # test2 = test2
    test2 = np.array([float(x) for x in test2.split(' ')])
    
    offset = test2[0]
    test2 = test2 - offset
    from scipy.stats import zscore
    test2 = zscore(test2)
    test2 = np.array([test2])
    
    X_train, y_train, X_test, y_test = classifier.load_data()
    X_train = X_train.reshape(len(X_train), len(X_train[0]), 1)
    
    test2 = test2.reshape( 20,  1)

    # fix offset to 0

    predictions = model.predict(np.array([test2]), batch_size=1)
    for prediction in predictions:
        print (prediction.max(), prediction.argmax())
    