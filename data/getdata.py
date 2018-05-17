#! /usr/bin/python3

import google.datalab.bigquery as bq

nyse = bq.Query.from_table(bq.Table('bingo-ml-1.market_data.nyse'), fields=['Date', 'Close']).execute().result().to_dataframe()

print(nyse)
nyse=nyse.sort_values(by='Date')
print (nyse)
#print(newlist)
nyse.to_csv('sorted.csv',sep=',', header=False, index=False)

