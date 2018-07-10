class Shapelet:

    def __init__(self, shapelet, index):
        self.start_index = index
        self.end_index = index + len(shapelet)
        self.shapelet = shapelet

    def __str__(self):
        return "this is a shapelet"
