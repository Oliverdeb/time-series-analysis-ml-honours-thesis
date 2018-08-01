from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
import pandas as pd
import numpy as np
import argparse

input_file = '../1000-20-30-20-38.csv'

def load_data(test_split = 0.2):
    print ('Loading data...')
    df = pd.read_csv(input_file)
    df['sequence'] = df['sequence'].apply(lambda x: [float(e) for e in x.split()[:20]])
    df = df.reindex(np.random.permutation(df.index))
    train_size = int(len(df) * (1 - test_split))

    X_train = np.array(df['sequence'].values[:train_size])
    y_train = np.array(df['target'].values[:train_size])
    X_test = np.array(df['sequence'].values[train_size:])
    y_test = np.array(df['target'].values[train_size:])
    # print (X_train[:10], y_train[:10], X_test[:10], y_test[:10])
    return pad_sequences(X_train, dtype='float64'), y_train, pad_sequences(X_test, dtype='float64'), y_test


def create_model(input_length):
    print ('Creating model...')
    model = Sequential()
    # model.add(Embedding(input_dim = 3000, output_dim = 5, input_length = input_length))
    # model.add(LSTM(output_dim=256, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True))
    # model.add(Dropout(0.5))
    # model.add(LSTM(output_dim=256, activation='sigmoid', inner_activation='hard_sigmoid'))
    # model.add(Dropout(0.5))

    # new way
    model.add(LSTM(128, recurrent_activation="hard_sigmoid", activation="sigmoid", return_sequences=True, input_shape=(20, 1)))
    model.add(Dropout(0.5))
    # model.add(LSTM(128,  recurrent_activation="hard_sigmoid", activation="sigmoid"))
    # model.add(Dropout(0.5))
    model.add(Dense(38, activation='sigmoid'))

    print ('Compiling...')
    # model.compile(loss='binary_crossentropy',
    #               optimizer='rmsprop',
    #               metrics=['accuracy'])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # model.summary()
    return model

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-load", help="load model from file", action="store_true", default=False)
    parser.add_argument("-save", help="create and save new model to file", action="store_true", default=False)

    args = parser.parse_args()

    if not args.load:
        X_train, y_train, X_test, y_test = load_data()
        print (X_train.shape )
        X_train = X_train.reshape(len(X_train), len(X_train[0]), 1)
        print (X_test.shape)
        X_test = X_test.reshape(len(X_test), len(X_test[0]), 1)
        model = create_model(len(X_train[0]))
        print ('Fitting model...')

        hist = model.fit(X_train, y_train, batch_size=64, epochs=500, validation_split = 0.1, verbose = 1)
        if args.save:
            model.save("2_LSTM_128_offset_model.h5")
        
        score, acc = model.evaluate(X_test, y_test, batch_size=1)

        print('Test score:', score)
        print('Test accuracy:', acc)
    else:
        print ("Loading model from file...")
        model = load_model("new_single_LSTM_256_offset_model.h5")
    pls = []
    # for line in open('../out.csv').readlines()[100:110]:
        # pls.append([float(x) for x in line.split(',')[1].split(' ')][:20])
    
    test = "1978.09 1953.03 1961.05 1952.29 1942.04 1969.41 1921.22 1951.13 1948.86 1913.85 1972.18 1988.87 1987.66 1940.51 1867.61 1893.21 1970.89 2035.73 2079.61 2096.92"
    test2 = "2270.75 2257.83 2238.83 2249.26 2249.92 2268.88 2263.79 2260.96 2265.18 2270.76 2262.53 2258.07 2262.03 2253.28 2271.72 2256.96 2259.53 2246.19 2241.35 2212.23"
    test2 = test
    test2 = np.array([float(x) for x in test2.split(' ')])
    offset = test2[0]
    test2 = test2 - offset
    test2 = np.array([test2])
    
    X_train, y_train, X_test, y_test = load_data()
    X_train = X_train.reshape(len(X_train), len(X_train[0]), 1)

    test2= test2.reshape( 20,  1)

    print( test2)


    # fix offset to 0

    predictions = model.predict(np.array([test2]), batch_size=32)
    for prediction in predictions:
        print (prediction.max(), prediction.argmax())
    