# Plate Waste Based Calorie Intake Predictor

In the hosptial setting, clinicians often rely on calorie counts (daily records of a patient's oral intake) to determine whether patients are ready to transition from one form of nutrition to another (i.e. removal of PEG with adequare oral intake). However, calorie counts are time consuming, and there are often inconsistencies in how unit staff records patient intake. 

Currently, there are applicattions that estimate the calorie and protein content of plates of food (before consumption). In the hospital setting, patients are plated standardized amounts of foods, and records are kept on the quantity of food each patient recevies. Therefore, we are able to levarege computer vision to estimate how much food is left on a patient's tray (plate-waste) to estimate their their meal time intake. 

In our project, we trained a computer vision model to estimate the calorie and protein content of a variety of plated foods. Data were obtained from Google's 2021 [Nutrition5k dataset](https://github.com/google-research-datasets/Nutrition5k), specifically using the set of overhead RGB.png images. These images included a robust diversity of both foodstuffs and plate fullness (i.e. a chicken entree, full salad, a few grapes on a plate). 

In the end were able to create a light-weight model (133.5 MB) based on the EfficientNet-B3 CNN. Our MAE during validation testing was  37.57 kcal and 4.23 g protein. In the context of calorie counts, this may be seen as clinically insignificant. However, further stress testing will need be done in real-world environments. 

## Setup

### Docker

**1. Clone the repo**
```bash
git clone https://github.com/raphi-l/plate-waste-calorie-count.git
cd plate-waste-calorie-count
```

**2. Download the model checkpoint**
```bash
pip install gdown
gdown "1DM2Lf9cRXgOqjXOgXnTiVFOQUEteuHtZ" -O models/efficientnet_b3_UF6_best.pt
```

**3. Build the image**
```bash
docker build -t plate-waste-calorie-count .
```

**4. Run the app**
```bash
docker run -p 8501:8501 -e NEBIUS_PLATE_API_KEY="your-nebius-api-key" plate-waste-calorie-count
```

**5. Open app in your local browser:**
`http://localhost:8501`

### Data Loading

Training images and data were obtained from Google's [Nutrition5k dataset](https://github.com/google-research-datasets/Nutrition5k)

To replicate real-world applications, we only trained our model on rgb.png overhead plated-food images. From the metadata csv's we extracted only `dish_id`, `total_calories` and `total_protein`.

Training and Test/Validation splits were performed using the `rgb_train_ids.txt` and `rgb_test_ids.txt` files from Google's repository. 

Data were pulled from the repository using:
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

The model will then give a summary of the inputed nutrition information, the estimated waste (kcal, protein) in the image, and the estiamted portion that the patient ate.

## Project Architecture

```
plate-waste-calorie-count/
├── README.md
├── Dockerfile
├── requirements.txt
├── mlflow.db
├── configs/
│   ├── best_model_report.yaml
│   ├── dish_data_config.yaml
│   └── model_config.yaml
├── models/
│   └── efficientnet_b3_UF6_best.pt
├── data/
│   └── raw/
│       ├── imagery/realsense_overhead/
│       │   └── dish_<id>/rgb.png        # 3,490 dishes
│       ├── metadata/
│       └── dish_ids/
│           └── splits/
│               ├── rgb_train_ids.txt
│               └── rgb_test_ids.txt
├── src/
│   ├── app.py
│   ├── build_model.py
│   ├── dataset.py
│   ├── evaluate.py
│   └── train.py
├── tests/
│   ├── test_app.py
│   ├── test_models.py
│   └── test_preprocessing.py
└── notebooks/
    └── plate_waste_calorie_count_colab_run.ipynb
```

## App Architecture
```
Clinician Input
      │
      ├─── Natural language    ────────────────▶   LLM Parser (Nebius AI)
      │    (i.e. "Patient was plated 350 kcal, 25 g protein) 
      │ 
      └─── Plate photo ──────▶ EfficientNet-B3
                                      │
                               Plate Waste:
                               Remaining kcal + protein on plate
                                      │
                               Patient Intake Calculation
                                (Plated amount - Plate Waste)
                                      │
                                LLM Responder (Nebius AI)
                                      │
                                Clinical Summary
```

## Methods
For our experiments, we iterated through linear-regression (sanity model) and EfficientNet-B0, B1, and B3, using the `pytorch` and `torchvision` libraries. Static hyperparameters were a learning rate 0.001 of for the model head, a learning rate of 1e-5 for backbone layers, and adam optimizer `EfficientNet_B<variant>_Weights.DEFAULT` pretrained weights were used for our baseline models.  

We used huberloss with a threshold of 50. For simplicity, we maintained this singular value for both calories and protein. To address the scale descepancy between calories and protein, we used penalty weights of [1.0 , 4.5] for calories and protein respectively prior to back-propagation. 

For our configurable hyperperameters, we explored unfreeezing either 0,4, or 6 backbone layers; dropout layers of 0.4 or 0.6. 

Using `mlflow` for experiment tracking, we ran a total of 7 differnt model configurations (refer to `configs/model_config.yaml`). Experiments were run using a `NVIDIA A100-SXM4-80GB` on Google Colab (refer to notebook: `notebooks/plate_waste_calorie_count_colab_run.ipynb`)

For metric reporting, we relied on Mean Absolute Error, R2, and MAPE. 

## Results Summary
It was found that EfficientNet-B3 with 6 non-batch normalization layers unfrozen offered the best results. 

In validation testing on novel images, our model acheived a Mean Absolute Error of 37.57 kcal and 4.23 g of protein. For our application, this approaching clinical insignificance. 

### Run Summary

| Parameter | Value |
|---|---|
| Checkpoint | `models/efficientnet_b3_UF6_best.pt` |
| Best Epoch | 54 |
| Validation Loss | 1204.27 |
| Targets | total_calories, total_protein |

### Hyperparameters

| Parameter | Value |
|---|---|
| Model | EfficientNet-B3 |
| Pretrained Weights | ImageNet (DEFAULT) |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Dropout | 0.6 |
| Layers Unfrozen | 6 |
| Huber Delta | 50.0 |

### Evaluation Metrics

| Metric | Calories | Protein |
|---|---|---|
| MAE | 37.57 kcal | 4.23 g |
| MAPE | 31.48% | 70.58% |
| R² | 0.892 | 0.728 |

## Limitations

All images found in the Nutrition5k datasets were images taken in highly controlled environments and of plates of food prior to consumption. Our current model may fail to perform in real world settings where post-meal plates will likely have mixed-up food items and garbage from the meal service.

This may be addressed by live-retraining of our model where manually verified post-meal food waste can be used to retrain the model.

## Future

For demonstration, we use a light-weight LLM `meta-llama/Meta-Llama-3.1-8B-Instruct` to parse caloried and protein plated for each patient. However, in real-world applications we would integrate our app with existing hospital food service managment systems (i.e. CBOARD) to directly pull patients' plated food data.

Our current Docker build has a content size of 3.32GB, largely attributed to the size of the torch library. In future builds, we will transition to ONNX for a prediction only library.

## 👤 Author

**Raphael** — Registered Dietitian & ML practitioner  
[HuggingFace](https://huggingface.co/raphi-l) · [GitHub](https://github.com/raphi-l)
