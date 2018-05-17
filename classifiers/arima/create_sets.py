#! /usr/bin/python3
from pandas import Series
series = Series.from_csv('sorted.csv', header=0)
split_point = int(len(series)*0.9)
dataset, validation = series[0:split_point], series[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv')
validation.to_csv('validation.csv')
