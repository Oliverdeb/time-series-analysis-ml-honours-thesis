import argparse, os
from re import sub
from keras.models import load_model
import numpy as np
from shapelets.shapelet_utils import shapelet_utils
from time import time
from shapelets.shapelet import Shapelet
from pandas import read_csv
from scipy.stats import zscore

class Predictor:
    def __init__(self, load_file=None, graph_file=None):
        if not load_file:
            print ("no file provided")
            exit(1)
        self.load_file = load_file
        self.graph_file = graph_file if graph_file else load_file.rpartition('model-')[2].replace('.h5', '.graph')
        self.graph_file = sub(r'reprocessed[0-9]+-*', '', self.graph_file)
        self.load_test_data()

    def load_test_data(self):
        """
        Load various information required. 

        - LSTM model from file
        - X_test and y_test that was used to train the LSTM model
        - Load in shapelets and shapelet dict from graph file

        """

        print ('\n\nLoading model...', self.load_file)

        # load model into memory
        try:
            self.model = load_model(self.load_file)
        except OSError:
            self.model = load_model('shapelets/' + self.load_file)

        # # test data corresponding to test data used to train model read in above
        print ('\nLoading test_data from model')
        test_name = 'test_data_output' + self.load_file
        from pickle import load
        try:
            in_dict = load(open(test_name,'rb'))
        except:
            in_dict = load(open('shapelets/'+test_name, 'rb'))
        self.in_dict = in_dict
        self.X_test, self.y_test = in_dict['X_test'], in_dict['y_test']

        # load shapelets and shapelet dict in from graph file in 
        print ('\nLoading shapelets and shapelet dict from', self.graph_file)
        try:
            load_graph_file = open(self.graph_file, 'rb')
        except FileNotFoundError:
            try:
                load_graph_file = open('shapelets/output/'+self.graph_file, 'rb')
            except FileNotFoundError:
                print ('%s file not found!' % self.graph_file)
                exit(1)

        in_dict = load(load_graph_file)
        self.shapelets, self.shapelet_dict, self.mse = in_dict['shapelets'], in_dict['shapelet_dict'], in_dict['mse']
        _dir = 'data/jse'
        self.datasets = {}
        for _file in ('StatisticsHistory-REDEFINE-2018-06-04.csv', 'StatisticsHistory-NASPERS-N-2018-06-04.csv'):
            dataset = read_csv(os.path.join(_dir, _file), usecols=[1])
            set_len = len(dataset.values)
            self.datasets[_file] = [x[0] for x in dataset.values.astype('float32')[int(set_len*0.7):]]

    def predict_price_from_shapes(self):
        """
        Predict prices from shapes selected manually, to compare to LSTM shape classifications
        """
        bars = ('-'*55)
        print ('\n%s\nPredicting price using MANUAL classification of test data\n%s\n' % (bars, bars))
        look_back = self.model.input_shape[1]
        shapelet_length = len(self.shapelets[0].std_shapelet)

        # loop through Naspers and Redefine datasets
        for _file, dataset in self.datasets.items():
            print ('='*len(_file))
            print ('MANUAL Testing %s' % _file.split('-')[1])
            print ('='*len(_file))
            _time = time()

            # create windows of size look back from the dataset
            X_test = [Shapelet(dataset[i:i+look_back], index=i) for i in range(len(dataset)) 
                    if len(dataset) - i >= shapelet_length
                    and np.std(dataset[i:i+look_back]) != 0]

            number_classifications_used = 0
            number_predictions_made = 0
            sumsquared_error = 0.0
            shapes_to_plot = []

            # loop through shape candidates formed on the dataset and manually select the best
            # shape from the list of shapelets for price predictions
            for i,shape in enumerate(X_test):
                print ('\r%.3f '%(i/len(X_test)), end='')

                # search through all shapelets for best match
                id = shapelet_utils.search_all(
                    shape,
                    self.shapelets[:self.in_dict['n_classes']],
                    self.shapelet_dict
                )

                # best match
                pred_shape = self.shapelet_dict[id]

                # if manually selected shape is not above threshold
                if not shapelet_utils.mse_dist(pred_shape.std_shapelet, shape.std_shapelet, self.mse):
                    continue

                # calculate mean and standard dev of last M points
                number_classifications_used += 1
                mean = np.mean(shape.shapelet)
                std = np.std(shape.shapelet)
                mse = 0
                start = shape.start_index + look_back
                end = shape.start_index + len(pred_shape.std_shapelet)

                # unstandardize shape using mean and standard dev of last M points
                unstandardized_prices = self.unstandardize_predictions(
                    dataset[start : end],
                    pred_shape.std_shapelet[look_back:],
                    mean,
                    std
                )

                # update SSE and mse
                sumsquared_error += unstandardized_prices.sum()
                number_predictions_made += len(unstandardized_prices)
                mse = unstandardized_prices.sum() / len(unstandardized_prices)
                                
                shapes_to_plot.append((
                    mse,
                    Shapelet(pred_shape.std_shapelet * std + mean, index=shape.start_index)
                ))


            # output stats about predictions and plot best 3 matches
            mse = sumsquared_error / number_predictions_made
            self.stats(_file, mse, len(X_test), number_classifications_used, number_predictions_made, _time)
            
            shapes_to_plot.sort(key=lambda x:x[0])
            shapes_to_give = [shapes_to_plot[0][1], shapes_to_plot[1][1], shapes_to_plot[2][1]]
            shapelet_utils.graph_shapes_on_series(shapes_to_give, dataset,  _file, lb=self.model.input_shape[1])
    
    
    def stats(self, _file, mse, n_candidates, n_class, n_pred, _time):   
        """
        Function that outputs stats about predictions made
        """     
        print ('total # shapelet candidates: %d' % (n_candidates))
        print ('total # shapelet classifications used: %d' % (n_class))
        print ('total # shapelet classifications NOT used: %d' % (n_candidates-n_class))
        print ('total # predictions made: %d' % (n_pred))
        print ('MSE: %.3f' % mse)
        print ('RMSE: %.3f' % np.sqrt(mse))
        print ('Time taken: %.3f' % (time() - _time))
        print ('='*len(_file))
        print()

    def compare_classification_accuracy_with_euclidean_distance(self):
        """
        Compares classification accuracy of LSTM model vs manually selection shapelet classes
        by distance
        """

        print ('Comparing model accuracy to manual shapelet selection')
        from sklearn.metrics import cohen_kappa_score

        X_test = self.X_test.reshape(len(self.X_test), len(self.X_test[0]))
        n_correct = 0
        manual_preds = []
        for x,y in zip(X_test, self.y_test):

            # search all classes manually based future shape
            target = shapelet_utils.search_classes(                
                Shapelet(std_shapelet=np.array(x)),
                self.shapelets[:self.in_dict['n_classes']],
                self.shapelet_dict,
                self.mse)
            manual_preds.append(target)

            if target == y:
                n_correct += 1

        # accuracy based on manual selection
        print ("Manual selection: %.3f " % (n_correct / len(X_test)) )
        
        # get classification predictions from LSTM model
        preds = self.model.predict(self.X_test)
        model_preds = [pred.argmax() for pred in preds]
        n_correct = 0
        for pred, y in zip(preds, self.y_test):
            if pred.argmax() == y:
                n_correct += 1

        print ("Model accuracy: %.3f" % (n_correct / len(preds)))

        print ('Cohens Kappa between manual selection and ground truth: ' , cohen_kappa_score(manual_preds, self.y_test))
        print ('Cohens Kappa between manual selection and model classifications: ' , cohen_kappa_score(manual_preds, model_preds))


    def predict_price_from_lstm(self):
        """
        Function that makes price predictions using shape classifications from the LSTM.
        """
        bars = ('-'*55)
        print ('\n%s\nPredicting price using LSTM classification of test data\n%s\n' % (bars, bars))
        look_back = self.model.input_shape[1]
        shapelet_length = len(self.shapelets[0].std_shapelet)

        # Predict on two datasets, Naspers and Redefine
        for _file, dataset in self.datasets.items():
            print ('='*len(_file))
            print ('LSTM Testing %s' % _file.split('-')[1])
            print ('='*len(_file))
            _time = time()
            
            # create windows of size look back from the dataset
            X_test_with_index = [
                (i,dataset[i:i+look_back]) for i in range(len(dataset)) 
                if len(dataset) - i >= shapelet_length
                and np.std(dataset[i:i+look_back]) != 0
            ]

            X_test = [snd for (fst,snd) in X_test_with_index]
            std_X_test = [zscore(x) for x in X_test]
            std_X_test = np.array(std_X_test).reshape(len(std_X_test), len(std_X_test[0]), 1)

            # get classifications from LSTM based on windows
            predictions = self.model.predict(std_X_test)

            number_predictions_made = 0
            number_classifications_used = 0
            sumsquared_error = 0.0
            shapes_to_plot = []

            # loop through predicted class labels and window
            for (index,x), prediction in zip (X_test_with_index, predictions):

                # probablity of class must be >= 70%.                
                if prediction.max() >= 0.7:
                    number_classifications_used += 1
                    pred_class_label = prediction.argmax()

                    instance_id = shapelet_utils.search_instance_of_class(
                        Shapelet(x),
                        self.shapelets[pred_class_label].id,
                        self.shapelet_dict,
                        self.mse)

                    shape = self.shapelet_dict[instance_id]

                    # calculate mean and standard dev of last M points
                    mean = np.mean(x)
                    std = np.std(x)
                    mse = 0
                    start = index + look_back
                    end = index + len(shape.std_shapelet)

                    # convert price back using standardized shape
                    unstandardized_prices = self.unstandardize_predictions(
                        dataset[start : end],
                        shape.std_shapelet[look_back:],
                        mean,
                        std
                    )

                    # update SSE, MSE and number of predictions made
                    sumsquared_error += unstandardized_prices.sum()
                    number_predictions_made += len(unstandardized_prices)
                    mse = unstandardized_prices.sum() / len(unstandardized_prices)
                                    
                    shapes_to_plot.append((
                        mse,
                        Shapelet(shape.std_shapelet * std + mean, index=index)
                    ))

            # output stats about predictions made and plot the first 3 best predictions
            # on the time series
            mse = sumsquared_error / number_predictions_made
            self.stats(_file, mse, len(X_test), number_classifications_used, number_predictions_made, _time)
            shapes_to_plot.sort(key=lambda x:x[0])
            shapes_to_give = [shapes_to_plot[0][1], shapes_to_plot[1][1], shapes_to_plot[2][1]]
            shapelet_utils.graph_shapes_on_series(shapes_to_give, dataset,  _file, lb=self.model.input_shape[1])
    
    def unstandardize_predictions(self, actual, standardized_shape, mean, std):
        """
        Unstandardize shape using mean and standard deviation and return the 
        squared difference between the predicted and actual price
        """
        predicted = standardized_shape * std + mean
        return (np.array(actual) - predicted) ** 2
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-load", help="load model from file", required=True)
    parser.add_argument('-graph', help='graph file', required=False)
    args = parser.parse_args()

    predict = Predictor(args.load, args.graph)
    predict.compare_classification_accuracy_with_euclidean_distance()
    # predict.predict_price_from_shapes()
    # predict.predict_price_from_lstm()

    # predict(args.load, args.graph)