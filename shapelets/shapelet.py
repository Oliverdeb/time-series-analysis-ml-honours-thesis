class Shapelet:

    def __init__(self, shapelet, index, quality=0):
        self.start_index = index
        self.shapelet = shapelet
        self.quality = quality

    def __lt__(self, other):
        return self.quality < other.quality

    def __gt__(self, other):
        return self.quality > other.quality
    
    def tocsv(self):
        s = str(self.shapelet[0])
        for e in self.shapelet[1:]:
            s += ' ' + str(e)
        return s
        
    def __str__(self):
        return "start index" + str(self.start_index) + " " +  str(self.shapelet)
