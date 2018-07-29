from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
import pandas as pd
import numpy as np

input_file = '../out.csv'

def load_data(test_split = 0.2):
    print ('Loading data...')
    df = pd.read_csv(input_file)
    df = df[:257]
    df['sequence'] = df['sequence'].apply(lambda x: [float(e) for e in x.split()])
    df = df.reindex(np.random.permutation(df.index))

    
    
    train_size = int(len(df) * (1 - test_split))

    X_train = df['sequence'].values[:train_size]
    y_train = np.array(df['target'].values[:train_size])
    X_test = np.array(df['sequence'].values[train_size:])
    y_test = np.array(df['target'].values[train_size:])

    return pad_sequences(X_train), y_train, pad_sequences(X_test), y_test


def create_model(input_length):
    print ('Creating model...')
    model = Sequential()
    model.add(Embedding(input_dim = 188, output_dim = 50, input_length = input_length))
    model.add(LSTM(output_dim=256, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(output_dim=256, activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(37, activation='sigmoid'))

    print ('Compiling...')
    # model.compile(loss='binary_crossentropy',
    #               optimizer='rmsprop',
    #               metrics=['accuracy'])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.summary()
    return model


X_train, y_train, X_test, y_test = load_data()

model = create_model(len(X_train[0]))

print ('Fitting model...')
hist = model.fit(X_train, y_train, batch_size=64, nb_epoch=10, validation_split = 0.1, verbose = 1)

score, acc = model.evaluate(X_test, y_test, batch_size=1)
print('Test score:', score)
print('Test accuracy:', acc)