
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense
from keras.models import Sequential
from EDDM import EDDM
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# convert an array of values into a dataset matrix
def create_dataset(dataset, end, start):
	dataX, dataY = [], []
	for i in range(start, end-look_back):
		dataX.append(dataset[i:(i+look_back),0])
	for i in range(start+look_back, end):
		dataY.append(dataset[i,0])
	return numpy.array(dataX), numpy.array(dataY)

# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
# load the dataset, Change DATASET.csv to the file you want to load, change usecols=[0] to correct coloumn for the dataset
dataframe = read_csv('DATASET.csv', usecols=[0], engine='python', skipfooter=0)
dataset = dataframe.values
dataset = dataset.astype('float32')
EDDM = EDDM()
driftCount = 0
drift = False
# Creatse datasets
look_back = 10 #window size change this to have a different Window size
endTrain = 100 #end point of first set of training data from dataset
startTrain = 0 #Start point of first set of training data from dataset
trainX, trainY = create_dataset(dataset, endTrain, startTrain) #reshape dataset training
startTest = endTrain-look_back #start of testing data from dataset
endTest = 120 #end of testing data from dataset
testX, testY = create_dataset(dataset, endTest, startTest) #reshape dataset testing
# create and fit Multilayer Perceptron model
model = Sequential()
model.add(Dense(12, input_dim=look_back, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=400, batch_size=2, verbose=2)
# Estimate model performance for each training and testing set
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))
#setting values for keep track of Accuracy, Percent error, drifts etc.
allPredictions = 0
allCorrect = 0
totalDrifts = 0
totalError = 0
currError = 0
#Check how much of the test predictions are correct, how many drifts there are and get percent error
def checkPredict(testX, testY):
  currentDataPoint= 0
  driftCount = 0
  totalPredictions = 0
  correctPredictions = 0
  currentError = 0
  #create test predictions
  testPredict = model.predict(testX)
  #loop through all the test predicitions and compare to actual results
  for y in testPredict:
	  if(y >= (testY[currentDataPoint]*0.98)):
		  if(y <= (testY[currentDataPoint]*1.02)):
			  #if the prediction is correct tell the EDDM this
			  drift = EDDM.set_input(True)
			  correctPredictions += 1
		  else:
			  #if it prediction is wrong tell the EDDM this
			  drift = EDDM.set_input(False)
	  else:
		  #if it prediction is wrong tell the EDDM this
		  drift = EDDM.set_input(False)
	  if(drift):
		  driftCount += 1
	  currentError += abs((y-testY[currentDataPoint])/testY[currentDataPoint])
	  currentDataPoint += 1
	  totalPredictions += 1
  return totalPredictions, correctPredictions, driftCount, currentError


currentTotal, currecntCorrect, currentDrifts,currError = checkPredict(testX,testY)
#add the current values to the total values
allPredictions += currentTotal
allCorrect += currecntCorrect
totalDrifts += currentDrifts
totalError += currError
#create new start and end for training and test data
startTrain += 20
endTrain = endTrain + 20
trainX, trainY = create_dataset(dataset, endTrain, startTrain)
startTest = endTrain-look_back
endTest = startTest + 20 + look_back
testX, testY = create_dataset(dataset, endTest, startTest)

#loop until the trainings last datapoint was the last datapoint in the set.
while(endTrain <= len(dataset)-1):
#train the model on the new data
  model.fit(trainX, trainY, epochs=400, batch_size=2, verbose=2)
  currentTotal, currecntCorrect, currentDrifts,currError = checkPredict(testX,testY)
  #add the current values to the total values
  allPredictions += currentTotal
  allCorrect += currecntCorrect
  totalDrifts += currentDrifts
  totalError += currError
  #Update the new start training data and end training data
  startTrain += 20
  endTrain = endTrain + 20
  if(endTrain>len(dataset)-1):
	  break
  trainX, trainY = create_dataset(dataset, endTrain, startTrain)
  #Update the testing data
  startTest = endTrain-look_back
  endTest = startTest + 20 + look_back
  if(endTest>len(dataset)-1):
  	  break
  testX, testY = create_dataset(dataset, endTest, startTest)
#print final results
print("Test drifts: ",totalDrifts, "Accuracy: ", (allCorrect/allPredictions)," Percent error: ",((totalError/allPredictions)*int(100))," Last tested: ",endTrain," total predictions: ",allPredictions)
