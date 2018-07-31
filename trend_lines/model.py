from trend_lines.simple_lstm import simple_lstm

class model:

    def __init__(self):
        self.slope_model = simple_lstm()

        self.duration_model = simple_lstm()

    def predict(self, slope_and_durations):
        pass

    