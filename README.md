# Plate Waste Based Calorie Intake Predictor

In the hosptial setting, clinicians often rely on calorie counts to determine whether patients are ready to transition from one form of nutrition to another (i.e. remove of PEG with adequare oral intake). However, calorie counts are often time consuming, and there are often in consistencies in how unit staff record patient intake. 

Currently, there are consumer grade applicattions that estimate the calorie and protein content of plates of food (before consumption). In the hospital setting, patients are plated standardized amounts of foods, and records are kept on the quantity of food each patient recevies. Therefore, we are able to levarege computer vision to estimate how much food is left on a patient's tray (plate-waste) to estimate their their meal time intake. 

Data were obtained from Google's 2021 Nutrition5k dataset, specifically using the set of overhead RGB.png images.

## Setup

### Docker
1. Pull/build the image
```bash
docker build -t plate-waste-calorie-count .
```

2. Run the app:
```bash
docker run -p 8501:8501 -e NEBIUS_PLATE_API_KEY="your-nebius-api-key" plate-waste-calorie-count
```

3. Open app in your local browser: `https://localhost:8501`

### Data Loading
Training images and data were obtained from Google's [Nutrition5k dataset](https://github.com/google-research-datasets/Nutrition5k)

To replicate real-world applications, we only trained our model on rgb.png overhead plated-food images. From the metadata csv's we extracted only `dish_id`,`total_calories` and `total_protein`

Data were pulled from Google's repository using:
```bash
mkdir -p data/raw/imagery data/raw/metadata data/raw/dish_ids

gcloud storage cp -r "gs://nutrition5k_dataset/nutrition5k_dataset/imagery/realsense_overhead/" data/raw/imagery/
gcloud storage cp -r "gs://nutrition5k_dataset/nutrition5k_dataset/metadata/" data/raw/
gcloud storage cp -r "gs://nutrition5k_dataset/nutrition5k_dataset/dish_ids/" data/raw/

# remove depth images — not used
find data/raw/imagery/ -name "depth_raw.png" -delete
find data/raw/imagery/ -name "depth_color.png" -delete
```

## App Use
Once loaded, users will be prompted to input the amount (kcal, protein) of food that was plated for each patient (i.e. `Patient was plated 350 kcal and 25 g protein`; or simply `350 kcal, 25 g protein`). 

Then users will be able to upload a picture of a patient's completed (or partially completed) food tray. 

The model will then .........

## App Architecture

## Results Summary

## Limitations
All images found in the Nutrition5k datasets were images taken in highly controlled environments and of plates of food prior to consumption. Our current model may fail to perform in real world settings where post-meal plates will likely have mixed-up food items and garbage from the meal service. 

This may be addressed by live-retraining of our model where manually verified post-meal food waste can be used to retrain the model. 

## Future 
For demonstration, we use a light-weight LLM `meta-llama/Meta-Llama-3.1-8B-Instruct` to parse caloried and protein plated for each patient. However, in real-world applications we would integrate our app with existing hospital food service managment systems (i.e. CBOARD) to directly pull patients' plated food data. 

Our current Docker build has a content size of 3.32GB, largely attributed to the size of the torch library. In future builds, we will transition to ONNX for a prediction only library. 

