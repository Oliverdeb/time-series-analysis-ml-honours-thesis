
import numpy
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from EDDM import EDDM
import os
global totalTrainRMSE
global totalTestRMSE
totalTestRMSE = 0
totalTrainRMSE = 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset, Change DATASET.csv to the file you want to load, change usecols=[0] to correct coloumn for the dataset
dataframe = pandas.read_csv('DATASET.csv', usecols=[0], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')
EDDM = EDDM()
driftCount = 0
drift = False
# convert an array of values into a dataset matrix
def create_dataset(dataset, end, start):
	dataX, dataY = [], []
	for i in range(start, end-look_back):
		dataX.append(dataset[i:(i+look_back),0])
	for i in range(start+look_back, end):
		dataY.append(dataset[i,0])
	return numpy.array(dataX), numpy.array(dataY)
#create datasets
look_back = 1 #window of only 1 value in the past
endTrain = 100 #end point of first set of training data from dataset
startTrain = 0 #Start point of first set of training data from dataset
trainX, trainY = create_dataset(dataset, endTrain, startTrain) #reshape dataset training
startTest = endTrain-look_back #start of testing data from dataset
endTest = 120 #end of testing data from dataset
testX, testY = create_dataset(dataset, endTest, startTest) #reshape dataset testing
# create and fit Multilayer Perceptron models
def create_model(trainX,trainY,testX,testY):
    newTrainX= trainX
    newTrainY= trainY
    newTestX= testX
    newTestY= testY
    model = Sequential()
    model.add(Dense(8, input_dim=look_back, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(newTrainX, newTrainY, epochs=400, batch_size=2, verbose=2)
    # Estimate model performance
    trainScore = model.evaluate(newTrainX, newTrainY, verbose=0)
    print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
    testScore = model.evaluate(newTestX, newTestY, verbose=0)
    print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))
    global totalTrainRMSE
    totalTrainRMSE += trainScore
    global totalTestRMSE
    totalTestRMSE += testScore
    return model

#setting values for keep track of Accuracy, Percent error, drifts etc.
allPredictions = 0
allCorrect = 0
totalDrifts = 0
currError = 0
totalError = 0

#Check how much of the test predictions are correct, how many drifts there are and get percent error
def checkPredict(testX, testY):
    currentDataPoint= 0
    driftCount = 0
    totalPredictions = 0
    correctPredictions = 0
    currentError = 0
    drift = False
	#Each model makes their own predicition
    testPredict1 = model1.predict(testX)
    testPredict2 = model2.predict(testX)
    testPredict3 = model3.predict(testX)
    testPredict4 = model4.predict(testX)
    for i in range(len(testY)):
		#final prediction is the average of all the models predictions
        prediction = (testPredict1[i]+testPredict2[i]+testPredict3[i]+testPredict4[i])/4
        if(prediction >= (testY[currentDataPoint]*0.98)):
            if(prediction <= (testY[currentDataPoint]*1.02)):
				#if the prediction is correct tell the EDDM this
                drift = EDDM.set_input(True)
                correctPredictions = correctPredictions + 1
            else:
				#if it prediction is wrong tell the EDDM this
                drift = EDDM.set_input(False)
        else:
			#if it prediction is wrong tell the EDDM this
            drift = EDDM.set_input(False)
		#If drift update counters
        if(drift):
            driftCount += 1
        currentError += abs((prediction-testY[currentDataPoint])/testY[currentDataPoint])
        currentDataPoint += 1
        totalPredictions += 1
    return totalPredictions, correctPredictions, driftCount, currentError

#create 4 models all trained on different data
model1 = create_model(trainX[:int(len(trainX)/4)],trainY[:int(len(trainY)/4)],testX[:int(len(testX)/4)],testY[:int(len(testY)/4)])
model2 = create_model(trainX[int(len(trainX)/4):int(len(trainX)/2)],trainY[int(len(trainY)/4):int(len(trainY)/2)],testX[int(len(testX)/4):int(len(testX)/2)],testY[int(len(testY)/4):int(len(testY)/2)])
model3 = create_model(trainX[int(len(trainX)/2):int(3*len(trainX)/4)],trainY[int(len(trainY)/2):int(3*len(trainY)/4)],testX[int(len(testX)/2):int(3*len(testX)/4)],testY[int(len(testY)/2):int(3*len(testY)/4)])
model4 = create_model(trainX[int(3*len(trainX)/4):],trainY[int(3*len(trainY)/4):],testX[int(3*len(testX)/4):],testY[int(3*len(testY)/4):])

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
	#train each model on their new data
  model1.fit(trainX[:int(len(trainX)/4)],trainY[:int(len(trainY)/4)], epochs=400, batch_size=2, verbose=2)
  model2.fit(trainX[int(len(trainX)/4):int(len(trainX)/2)],trainY[int(len(trainY)/4):int(len(trainY)/2)], epochs=400, batch_size=2, verbose=2)
  model3.fit(trainX[int(len(trainX)/2):int(3*len(trainX)/4)],trainY[int(len(trainY)/2):int(3*len(trainY)/4)], epochs=400, batch_size=2, verbose=2)
  model4.fit(trainX[int(3*len(trainX)/4):],trainY[int(3*len(trainY)/4):], epochs=400, batch_size=2, verbose=2)
  currentTotal, currecntCorrect, currentDrifts,currError = checkPredict(testX,testY)
  #add the current values to the total values
  allPredictions += currentTotal
  allCorrect += currecntCorrect
  totalDrifts += currentDrifts
  totalError += currError
  #update training Data
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

print("Test drifts: ",totalDrifts, "Accuracy: ", (allCorrect/allPredictions)," Percent error: ",((totalError/allPredictions)*int(100))," Last tested: ",endTrain," total predictions: ",allPredictions)
