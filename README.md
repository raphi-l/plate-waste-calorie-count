# Plate Waste Based Calorie Intake Predictor

In the hosptial setting, clinicians often rely on calorie counts to determine whether patients are ready to transition from one form of nutrition to another (i.e. remove of PEG with adequare oral intake). However, calorie counts are often time consuming, and there are often in consistencies in how unit staff record patient intake. 

Currently, there are consumer grade applicattions that estimate the calorie and protein content of plates of food (before consumption). In the hospital setting, patients are plated standardized amounts of foods, and records are kept on the quantity of food each patient recevies. Therefore, we are able to levarege computer vision to estimate how much food is left on a patient's tray (plate-waste) to estimate their their meal time intake. 

Data were obtained from Google's 2021 Nutrition5k dataset, specifically using the set of overhead RGB.png images.

## Limitations
All images found in the Nutrition5k datasets were images taken in highly controlled environments and of plates of food prior to consumption. Our current model may fail to perform in real world settings where post-meal plates will likely have mixed-up food items and garbage from the meal service. 

This may be addressed by live-retraining of our model where manually verified post-meal food waste can be used to retrain the model. 


