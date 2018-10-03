# Predicting financial time series by use of shapelets and trend lines while exploring the notion of concept drift within

## Aim
This was an honours thesis project. I worked on the part of the project that aimed at extracting recurring patterns/shapes from time series data for prediction. [Robby](https://github.com/robertbodley) worked on the trendline portion of the project. [Andre](https://github.com/AndreVent) explored the notion of concept drift and methods for detection and avoiding drift.

### Shapelets:
#### Intro
The inspiration for this project came from the work done by Lines et al. on the [Shapelets Transform](http://wan.poly.edu/KDD2012/docs/p289.pdf). Ye et al. introduced the [notion of shapelets](http://alumni.cs.ucr.edu/~lexiangy/Shapelet/kdd2009shapelet.pdf) in 2009 as a new data mining primitive.

#### Process
1. Extract variable length shapelets from the datasets of [35 stocks](./data/jse/). See below images for examples of the extracted shapelets classes and an illustration of the classes extracted.
2. Train a LSTM based classifier on the extracted shapelet classes.
3. Predict prices by inputting "future" sequences into the LSTM so that it can classify which pattern/shapelet the sequence looks like. Use the standardized version of that classified shape to predict prices by unstandardizing using the mean and standard deviation of the "future" sequence. See prediction images below.


#### Plot of shapelet classes

Extracted classes on the source dataset             |  Extracted classes (standardized shapes)
:-------------------------:|:-------------------------:
![](larger_classes_series_1.png)  |  ![](class_len_21_mse04.png)



#### Predicting 1 day ahead
<p align="center">
<img align="center" width="75%" height="75%"  src="redefine_manual_best.png">
</p>

#### Predicting 7 days ahead
<p align="center">
<img align="center" width="75%" height="75%"  src="naspers_lstm_BEST.png">
</p>

### Concept Drift:
#### Intro
Concept drift is when the underlying data changes due to some unknown or unforeseen cause, making it so your machine learning models predictions become less accurate. Machine learning models must be able to identify and deal with multiple changing concepts as time goes on.

#### Process
Two concept drift detection methods implemented were the Page-Hinkley (PH) and Early Drift Detection Method (EDDM) detectors. These were paired with three mitigation strategies: 
1. The fixed sliding window, used a fixed number of data points to make a prediction.
2. The adjusting sliding window, used varying number of data points depending on if drifts were detected or not to make a prediction.
3. An ensemble approach, train four models on their own set of data and take a averaged prediction.

#### Final results:
<p align="center">
<img align="center"   src="FinalResults.png">
</p>
