from trend_lines.point import Point

class Line:
    
    def __init__(self, slope, duration, start, c):
        self.slope = slope
        self.duration = duration
        self.start = start
        self.c = c

    def lineArray(self):
        line = []
        for x in range(self.duration+1):
            line.append(Point(self.start+x, float(self.c) + float(self.slope*x), self.start+x))
        return line

    def __str__(self):
        return str(self.slope) + " : " + str(self.start) + " : " + str(self.duration+self.start) + " : " + str(self.c)
