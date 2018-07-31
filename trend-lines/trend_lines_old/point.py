
class Point:

    def __init__(self, time, price, index):
        self.time = time
        self.price = price
        self.index = index

    def __str__(self):
        return str(self.time) + " : " + str(self.price) + " : " + str(self.index)
