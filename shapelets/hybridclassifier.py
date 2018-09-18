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
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')


class LSTMClassifier:

    def __init__(self, input_file, lb):
        self.input_file = input_file
        self.look_back = lb

    def load_data(self, std, print_labels=False, test_split = 0.33):
        """
        Function that loads in the dataset, standardizes and creates the train/test split 
        """
        print ('Loading data...')
        df = pd.read_csv(self.input_file)

        # convert to float
        df['sequence'] = df['sequence'].apply(lambda x: [float(e) for e in x.split()[:self.look_back]])
        
        # standardize each shapelet individually
        if std is False:
            print ("Standardizing input data")
            df['sequence'] = df['sequence'].apply(zscore)
        df = df[df.sequence.map(lambda x: False in np.isnan(x))]

        if print_labels:
            print(df.groupby(['target']).count())

        self.n_classes = df['target'].values[-1] + 1

        # randomize ordering        
        new_index = list(df.index)
        np.random.shuffle(new_index)
        df = df.reindex(new_index)

        if print_labels:
            print (df)

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
        """
        Function that creates and compiles the LSTM model
        """
        print ('Creating model...')
        with tf.device('/cpu:0'):
            model = Sequential()

            # one input LSTM layer
            model.add(LSTM(
                512,
                activation="sigmoid",
                input_shape=(self.look_back, 1),
                recurrent_dropout=0.3,
                return_sequences=True))
            model.add(Dropout(0.5))

            # one hidden LSTM layer
            model.add(LSTM(
                256,
                recurrent_dropout=0.1,
                activation="sigmoid",))
            model.add(Dropout(0.5))

            # One output layer
            model.add(Dense(self.n_classes, activation='softmax'))
        print ('Compiling...')

        opt = 'adam'
        if n != 0:
            # compile parallel GPU model for specified number of GPUs
            parallel_model = multi_gpu_model(model, gpus=n)
            parallel_model.compile(
                loss='sparse_categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy']
            )
            return model, parallel_model
        else:
            # compile normal model
            model.compile(
                loss='sparse_categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy']
            )
            return model

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-load", help="load model from file", default=None)
    parser.add_argument("-gpu", help="number of gpus to run on", required=True)
    parser.add_argument("-batch", help="batch size")
    parser.add_argument("-epochs", help="epochs")
    parser.add_argument("-std", help="whether to NOT standardize in preprocessing step", action='store_true')
    parser.add_argument("-f", help="filename", required=True)
    parser.add_argument("-lb", help="lookback", required=True)

    args = parser.parse_args()

    file_name = args.f if args.f else 'std_186892-20-20-0.7-1256.csv'
    input_file = 'output/' + file_name

    classifier = LSTMClassifier(input_file, int(args.lb))

    if args.load is None:
        X_train, y_train, X_test, y_test = classifier.load_data(args.std, True)
        n_gpus = int(args.gpu)
        print ('Fitting model...')
            
        if n_gpus != 0:
            original_model, parallel_model = classifier.create_model(n_gpus)
            model = parallel_model
        else:
            original_model = classifier.create_model(n_gpus)
            model = original_model
            n_gpus = 1

        batch_size = 32 if not args.batch else int(args.batch)
        print ("\n\n\n",'='*40,"\nTraining with batch size: %d" % batch_size, sep='')
        print ("Training on shapes with LOOKBACK OF %d\n" % classifier.look_back,'='*40,sep='')
        epochs = 600 if not args.epochs else int(args.epochs)
        
        history = model.fit(
            X_train,
            y_train,
            batch_size=batch_size*n_gpus,
            epochs=epochs,
            validation_split = 0.1,
            verbose = 1
        )

        model.summary()
        model_name = "{0}_classes_batchsize={1}_epochs_{2}_model-{3}.h5".format (classifier.n_classes, batch_size, epochs, file_name.replace('.csv', ''))
        original_model.save(model_name)

        from pickle import dump 
        
        dump({
            'X_test': X_test,
            'y_test': y_test,
            'n_classes' : classifier.n_classes,
        }, open('test_data_output'+model_name, 'wb'))
        score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)

        print('Test score:', score)
        print('Test accuracy:', acc)

        # various classification metrics
        from sklearn.metrics import classification_report, f1_score, cohen_kappa_score
        preds = [pred.argmax() for pred in model.predict(X_test, batch_size=batch_size)]
        l = ['class ' + str(x) for x in range(classifier.n_classes)]
        print(classification_report(y_test, preds, target_names=l))
        print ('cohens kappa:', cohen_kappa_score(y_test, preds))
        print ('avg f1:', f1_score(y_test, preds, average='micro'))


        # graph loss and accuracy 
        import matplotlib.pyplot as plt
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        # plt.show()
        plt.savefig('acc.png', bbox_inches='tight')
        plt.clf()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        # plt.show()
        plt.savefig('loss.png', bbox_inches='tight')

    else:
        print ("Loading model from file...")
        model = load_model(args.load)
    
    if args.load:
        # load test data in if testing model from file
        X_train, y_train, X_test, y_test = classifier.load_data(args.std)
        X_train = pad_sequences(X_train, dtype='float64')
        X_train = X_train.reshape(len(X_train), len(X_train[0]), 1)
    
    predictions = model.predict(X_test)

    # assess model accuracy
    print ('custom test:\n')
    n_correct = [1 for pred,y in zip(predictions, y_test) if pred.argmax() == y]
    acc = len(n_correct) / len(predictions)
    print('Custom Test accuracy:', acc)