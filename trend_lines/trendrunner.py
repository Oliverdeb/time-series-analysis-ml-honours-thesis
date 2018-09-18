

class Trendrunner:

    def run(self, mse=None):
        import time
        # from matplotlib.pylab import gca, figure, plot, subplot, title, xlabel, ylabel, xlim, show
        from matplotlib.pylab import figure
        # from matplotlib.lines import Line2D
        from trend_lines.segment import slidingwindowsegment, bottomupsegment, topdownsegment
        from trend_lines.fit import interpolate, sumsquared_error
        from trend_lines.wrappers import stats, convert_to_slope_duration, draw_plot, draw_segments
        
        with open("data/snp2.csv") as f:
            # with open("example_data/16265-normalecg.txt") as f:
            file_lines = f.readlines()

        data = [float(x.split("\t")[0].strip()) for x in file_lines]

        def dump_csv(name):
            with open('%s.csv' % name, 'w') as f:
                f.write('slope,duration\n')
                for slope, duration in convert_to_slope_duration(segments):
                    f.write(','.join(("%.2f" % slope, "%d" % duration)) + "\n")
        if mse is None:
            print("GIMME DA MSE?")
            exit(1)
        max_error = mse

        # sliding window with simple interpolation
        name = "Sliding window with simple interpolation"
        figure()
        start = time.time()
        segments = slidingwindowsegment(
            data, interpolate, sumsquared_error, max_error)
        stats(name, max_error, start, segments, data)
        draw_plot(data, name)
        draw_segments(segments)
        dump_csv('sliding-window')

        # bottom-up with  simple interpolation
        name = "Bottom-up with simple interpolation"
        figure()
        start = time.time()
        segments = bottomupsegment(
            data, interpolate, sumsquared_error, max_error)
        stats(name, max_error, start, segments, data)
        draw_plot(data, name)
        draw_segments(segments)
        dump_csv('bottom-up')

        
        # top-down with  simple interpolation
        name = "Top-down with simple interpolation"
        figure()
        start = time.time()
        segments = topdownsegment(
            data, interpolate, sumsquared_error, max_error)
        stats(name, max_error, start, segments, data)
        draw_plot(data, name)
        draw_segments(segments)

        # dump_csv('top-down')

        # show()

    def train_baby_train(self, epochs, batch_size, point_data_file, look_back):

        from trend_lines.simple_mlp_point_data import Simple_mlp_point_data
        from trend_lines.simple_lstm_point import simple_lstm
        from trend_lines.complex_lstm_trend import complex_lstm
        from trend_lines.complex_mlp_trend import complex_mlp

        names = ['top-down', 'bottom-up', 'sliding-window']

        params = (batch_size, epochs, look_back)

        print('\n\nRUNNING %s on MLP-POINT-DATA model batch size=%d, epochs=%d' %
              (point_data_file, batch_size, epochs))

        Simple_mlp_point_data(point_data_file, *params).run()
        print('\n\nRUNNING %s on LSTM-POINT-DATA model batch size=%d, epochs=%d' %
              (point_data_file, batch_size, epochs))

        simple_lstm(point_data_file, *params).run()

        # exit()
        for name in names:
            
            print('\n\nRUNNING %s on MLP-TREND model batch size=%d, epochs=%d' %
                  (name, batch_size, epochs))

            complex_mlp(point_data_file, "%s.csv" % name, *params).run()

            print('\n\nRUNNING %s on LSTM-TREND  model batch size=%d, epochs=%d' %
                  (name, batch_size, epochs))

            complex_lstm(point_data_file, "%s.csv" % name, *params).run()
            # exit()
