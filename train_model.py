import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from deap import base, creator, tools, algorithms
import joblib
import warnings
import random
import json
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 1. LOAD & PREPROCESS DATA
# ─────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("data/car data.csv")

print(f"Dataset shape: {df.shape}")
print(df.head())

# Drop rows with missing values
df.dropna(inplace=True)

# Feature engineering
df['Car_Age'] = 2024 - df['Year']

# Encode categorical columns
le_fuel = LabelEncoder()
le_seller = LabelEncoder()
le_trans = LabelEncoder()

df['Fuel_Type'] = le_fuel.fit_transform(df['Fuel_Type'])
df['Selling_type'] = le_seller.fit_transform(df['Selling_type'])
df['Transmission'] = le_trans.fit_transform(df['Transmission'])

# Drop columns not needed
df.drop(['Car_Name', 'Year'], axis=1, inplace=True)

# Features and target
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Convert all to float to avoid XGBoost issues
X = X.astype(float)

feature_names = list(X.columns)
print(f"\nFeatures: {feature_names}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─────────────────────────────────────────
# 2. GENETIC ALGORITHM FEATURE SELECTION
# ─────────────────────────────────────────
print("\nRunning Genetic Algorithm for Feature Selection...")

N_FEATURES = X_train.shape[1]

if hasattr(creator, "FitnessMax"):
    del creator.FitnessMax
if hasattr(creator, "Individual"):
    del creator.Individual

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_bool, n=N_FEATURES)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def eval_features(individual):
    selected = [i for i, bit in enumerate(individual) if bit == 1]
    if len(selected) == 0:
        return (0.0,)
    X_sel = X_train.iloc[:, selected].astype(float)
    model = XGBRegressor(n_estimators=50, random_state=42, verbosity=0)
    scores = cross_val_score(model, X_sel, y_train, cv=3, scoring='r2')
    return (scores.mean(),)

toolbox.register("evaluate", eval_features)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

random.seed(42)
population = toolbox.population(n=30)
NGEN = 20

print("Generation | Best R² Score | Features Selected")
print("-" * 50)

best_individual = None
best_fitness = -999
convergence_history = []

for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.7, mutpb=0.2)
    fits = list(map(toolbox.evaluate, offspring))
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

    top = tools.selBest(population, k=1)[0]
    selected_count = sum(top)
    print(f"Gen {gen+1:>3}      | {top.fitness.values[0]:.4f}        | {selected_count}")

    if top.fitness.values[0] > best_fitness:
        best_fitness = top.fitness.values[0]
        best_individual = list(top)

    convergence_history.append({
        'generation': gen + 1,
        'best_r2': top.fitness.values[0],
        'overall_best_r2': best_fitness
    })

selected_indices = [i for i, bit in enumerate(best_individual) if bit == 1]
selected_features = [feature_names[i] for i in selected_indices]
print(f"\nBest Features Selected by GA: {selected_features}")
print(f"Best CV R² Score: {best_fitness:.4f}")

# ─────────────────────────────────────────
# 3. TRAIN FINAL MODEL
# ─────────────────────────────────────────
print("\nTraining final XGBoost model...")

X_train_sel = X_train[selected_features].astype(float)
X_test_sel = X_test[selected_features].astype(float)

model = XGBRegressor(n_estimators=200, learning_rate=0.05,
                     max_depth=6, random_state=42, verbosity=0)
model.fit(X_train_sel, y_train)

y_pred = model.predict(X_test_sel)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nFinal Model Performance:")
print(f"  MAE  : {mae:.4f} Lakhs")
print(f"  R²   : {r2:.4f}")

# ─────────────────────────────────────────
# 4. SAVE MODEL & METADATA
# ─────────────────────────────────────────
joblib.dump(model, "models/car_price_model.pkl")
joblib.dump(selected_features, "models/selected_features.pkl")
joblib.dump(feature_names, "models/all_features.pkl")

# Feature importance from trained model
feature_importance = {
    feat: float(imp) for feat, imp in zip(selected_features, model.feature_importances_)
}

results = {
    'selected_features': selected_features,
    'best_r2_cv': best_fitness,
    'test_mae': mae,
    'test_r2': r2,
    'convergence_history': convergence_history,
    'feature_importance': feature_importance
}
with open("outputs/results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nModel saved to models/car_price_model.pkl")
print("Training complete!")