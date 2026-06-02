# XAI on Cultural Heritage Apps 🏛️

An Explainable AI project that analyzes Google Play Store data for
Cultural Heritage applications — predicting app reviews using regression
and classifying app characteristics using machine learning, with full
model interpretability pipelines.

## Problem statement
Cultural heritage apps play a key role in digital preservation and
public engagement. This project uses ML to predict the number of reviews
an app receives and classify app types, while explaining model decisions
using Explainable AI techniques on real Google Play Store data.

## Dataset

| Feature | Description |
|---------|-------------|
| Application Name | Name of the heritage app |
| Application Description | Full app description text |
| Number of Reviews | Total user reviews (regression target) |
| Overall Rate | Average user rating |
| First Page Reviews | Text of first page user reviews |
| Number of Days Since Last Update | Recency of updates |
| App Size | Size of the app in MB |
| App Installs | Number of installs |
| Version Required | Minimum Android version |
| Last Update | Date of last update |

- **Total records:** 992
- **Source:** Google Play Store — Cultural Heritage category

## Project workflow

```
1. Data Loading & Column Cleaning (drop unnamed columns)
2. Target Detection — Number of Reviews (regression)
3. Text Preprocessing — TF-IDF on app name + description
4. Feature Engineering:
   - Numeric: median imputation + StandardScaler
   - Categorical: OneHotEncoder
   - Text: TF-IDF (bigrams, max 200 features)
5. Dimensionality Reduction — TruncatedSVD (100 components)
6. Regression — predict Number of Reviews
7. Classification — classify app type/category
8. Model Evaluation & Visualization
9. Model Saving (.joblib)
```

## Models used

### Regression (predicting Number of Reviews)

| Model | Metric |
|-------|--------|
| Linear Regression | MSE, R² |
| Random Forest Regressor (200 trees) | MSE, R² |

### Classification (predicting app category)

| Model | Metric |
|-------|--------|
| Logistic Regression | Accuracy, F1 (weighted) |
| Random Forest Classifier (200 trees) | Accuracy, F1 (weighted) |

## Evaluation metrics

| Task | Metrics |
|------|---------|
| Regression | Mean Squared Error (MSE), R² Score |
| Classification | Accuracy, F1-Score (weighted), Confusion Matrix, ROC-AUC |

## Visualizations
- Feature distribution histograms
- Correlation heatmap
- Actual vs Predicted scatter plot (Random Forest Regressor)
- Confusion Matrix (Random Forest Classifier)
- ROC Curve (binary classification)

## Preprocessing pipeline

```
Text columns (Name + Description)
      ↓
  TF-IDF (200 features, bigrams)
      ↓
ColumnTransformer (Numeric + Categorical + Text)
      ↓
TruncatedSVD (100 components)
      ↓
Train/Test Split (80/20)
      ↓
ML Models
```

## Saved models

| File | Description |
|------|-------------|
| `linear_regression_model.joblib` | Linear Regression model |
| `random_forest_regressor.joblib` | Random Forest Regressor |
| `logistic_classification_model.joblib` | Logistic Regression Classifier |
| `random_forest_classifier.joblib` | Random Forest Classifier |
| `label_encoder_for_class.joblib` | Label Encoder for classes |

## How to run

```bash
git clone https://github.com/Mokshitha08/EXPLAINABLE-AI
cd EXPLAINABLE-AI
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

Then open the notebook in Jupyter or Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mokshitha08/EXPLAINABLE-AI/)

## Tools & libraries

`Python` · `pandas` · `NumPy` · `scikit-learn` · `matplotlib` ·
`seaborn` · `joblib` · `TF-IDF` · `TruncatedSVD` · `Jupyter Notebook`

## Key insights
- App description text is a strong predictor of review count
- Random Forest outperforms Linear Regression on sparse, high-dimensional app data
- TruncatedSVD effectively reduces TF-IDF features while preserving variance
- Recent app updates correlate positively with higher install counts

