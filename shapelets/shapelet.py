class Shapelet:

    def __init__(self, shapelet, index, quality=0):
        self.start_index = index
        self.shapelet = shapelet
        self.quality = quality

    def __lt__(self, other):
        return self.quality < other.quality

    def __gt__(self, other):
        return self.quality > other.quality
    
    def to_csv_offset_0(self):
        offset = self.shapelet[0]
        self.shapelet = self.shapelet - offset
        csv = "0"

        for elem in self.shapelet[1:]:
            csv += ' ' + str(elem)
        return csv

    def to_csv(self):
        csv = str(self.shapelet[0])
        for elem in self.shapelet[1:]:
            csv += ' ' + str(elem)
        return csv
        
    def __str__(self):
        return "start index" + str(self.start_index) + " " +  str(self.shapelet)
