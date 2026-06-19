# Car Valuation Engine
GA-optimized XGBoost model for used car price prediction.

![Python](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![NIC](https://img.shields.io/badge/NIC-29101-orange)

**NIC 29101 — Automotive Engineering**


---

## Overview

This project predicts the resale price of a used car using a machine learning pipeline where the input features are selected automatically by a **Genetic Algorithm**, rather than chosen manually. The selected features are then used to train an **XGBoost regression model**.

The goal was to demonstrate a practical application of evolutionary computation (the core focus of the internship) within an automotive engineering use case (NIC 29101).

---

## Problem statement

Used car pricing is typically estimated manually by sellers or dealers, often inconsistently. This project automates that estimation using historical sales data, so a fair price can be predicted from a car's specifications alone.

---

## Methodology

**1. Data**
- 301 used car listings: brand, year, present price, mileage, fuel type, seller type, transmission, ownership history
- Source: public used-car dataset (Kaggle)

**2. Feature engineering**
- Derived `Car_Age` from manufacturing year
- Label-encoded categorical fields (fuel type, seller type, transmission)

**3. Feature selection — Genetic Algorithm (DEAP)**
- Population of 30 candidate feature subsets, evolved over 20 generations
- Fitness function: 3-fold cross-validated R² of an XGBoost model trained on each subset
- Selection: tournament selection, two-point crossover, bit-flip mutation
- The algorithm converged on its best feature subset by generation 6 and remained stable afterward

**4. Model training**
- XGBoost Regressor trained on the GA-selected feature subset
- 80/20 train-test split

**5. Verification layer**
- Uploaded car images are checked using a pretrained MobileNetV2 (ImageNet) classifier before a prediction is allowed, to confirm the image is actually of a vehicle

---

## Results

| Metric | Value |
|---|---|
| Test R² | 0.9656 |
| Test MAE | 0.54 Lakhs |
| GA best cross-validated R² | 0.9128 |
| Generations to convergence | 6 |

The GA converged quickly and stably, indicating the selected feature subset is a genuine optimum for this dataset rather than a noisy artifact.

---

## Application

A Streamlit dashboard with three views:

- **Predict price** — input car specifications and an image; returns a predicted selling price with a price range and a table of similar cars from the dataset
- **Model performance** — test metrics, GA convergence plot, selected features, feature importance
- **Data insights** — dataset distributions and relationships

---

## Tech stack

- Python, pandas, scikit-learn
- XGBoost — regression model
- DEAP — genetic algorithm framework
- TensorFlow / MobileNetV2 — image verification
- Streamlit — application interface
- Plotly — interactive charts

---

## Project structure

```
car-price-prediction/
├── data/
│   └── car data.csv
├── models/
│   ├── car_price_model.pkl
│   ├── selected_features.pkl
│   └── all_features.pkl
├── outputs/
│   └── results.json
├── train_model.py
├── app.py
├── requirements.txt
└── README.md
```

---

## How to run

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

python train_model.py          # trains the model, saves to models/ and outputs/
streamlit run app.py           # launches the dashboard
```

---

## Author

Srilakshmi K
B.E. Information Science Engineering, JAIN University

