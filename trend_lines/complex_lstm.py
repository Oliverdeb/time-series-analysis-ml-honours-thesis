from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
import pandas as pd
import numpy as np
import argparse

class complex_lstm:
    def __init__(self, file_name):
        pass

    def load_data(self, file_name, test_split = 0.2):
        print ('Loading data...')
        df = pd.read_csv(file_name)
        dataset = df.values.astype('float32')

        x, y = self.create_dataset(dataset, 40)
        train_size = int(len(dataset) * (1 - test_split))

        X_train = np.array(x[:train_size])
        y_train = np.array(y[:train_size])
        X_test = np.array(x[train_size:])
        y_test = np.array(y[train_size:])
        # print (X_train[:10], y_train[:10], X_test[:10], y_test[:10])
        return X_train, y_train, X_test, y_test

    def create_dataset(self, dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back)]
            dataX.append(a)
            dataY.append(dataset[i + look_back])
        return np.array(dataX), np.array(dataY)

    def create_model(self):
        print ('Creating model...')
        self.model = Sequential()
        # model.add(Embedding(input_dim = 3000, output_dim = 5, input_length = input_length))
        # model.add(LSTM(output_dim=256, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True))
        # model.add(Dropout(0.5))
        # model.add(LSTM(output_dim=256, activation='sigmoid', inner_activation='hard_sigmoid'))
        # model.add(Dropout(0.5))

        # new way
        self.model.add(LSTM(256))#, input_shape=(40, 2)))
        self.model.add(Dropout(0.5))
        # model.add(LSTM(128,  recurrent_activation="hard_sigmoid", activation="sigmoid"))
        # model.add(Dropout(0.5))
        self.model.add(Dense(2))

        print ('Compiling...')
        # model.compile(loss='binary_crossentropy',
        #               optimizer='rmsprop',
        #               metrics=['accuracy'])
        # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', )
        self.model.compile(loss='mean_squared_error', optimizer='adam')

        # model.summary()
        return self.model

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-load", help="load model from file", action="store_true", default=False)
    parser.add_argument("-save", help="create and save new model to file", action="store_true", default=False)

    args = parser.parse_args()
    file_name = '../slope_dur.csv'
    lstm = complex_lstm(file_name)

    if not args.load:
        X_train, y_train, X_test, y_test = lstm.load_data(file_name)
        print (X_train.shape )
        # X_train = X_train.reshape(len(X_train), len(X_train[0]), 1)
        print (X_test.shape)
        # X_test = X_test.reshape(len(X_test), len(X_test[0]), 1)
        model = lstm.create_model()
        print ('Fitting model...')

        hist = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_split = 0.1, verbose = 1)
        if args.save:
            model.save("trend_comp_lstm-400-40.h5")
        
        score = model.evaluate(X_test, y_test, batch_size=1)

        print('Test score:', score)
        print('Test accuracy:', 1)
    else:
        print ("Loading model from file...")
        model = load_model("trend_comp_lstm-400-40.h5")
    pls = []
    # for line in open('../out.csv').readlines()[100:110]:
        # pls.append([float(x) for x in line.split(',')[1].split(' ')][:20])
    
    
    test = open(file_name).readlines()       
    test2 = []
    ind = 742
    for line in test[ind:ind+40]:
        test2.append(np.array([float(x) for x in line.split(',')]))
    X_train, y_train, X_test, y_test = lstm.load_data(file_name)
    score = model.evaluate(X_test, y_test, batch_size=1)

    print('Test score:', score)
    # X_train, y_train, X_test, y_test = load_data()
    # X_train = X_train.reshape(len(X_train), len(X_train[0]), 1)

    # test2= test2.reshape( 20,  1)

    # print( test2)


    # fix offset to 0
    print(test2)
    predictions = model.predict(np.array([test2]), batch_size=32)
    print (predictions)
    