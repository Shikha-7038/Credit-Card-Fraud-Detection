# 💳 Credit Card Fraud Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📌 Overview

An end-to-end machine learning system that detects fraudulent credit card transactions using imbalanced classification techniques. This project demonstrates a complete data science workflow from synthetic data generation to model deployment.

## 🎯 Problem Statement

Credit card fraud costs billions annually. Banks need systems that can:
- **Detect fraud** in near real-time
- **Minimize false positives** (don't block good customers)
- **Handle class imbalance** (fraud is rare - <1% of transactions)

## 🚀 Solution

This project builds a complete fraud detection pipeline:
Transaction → Preprocessing → Feature Engineering → Model → Prediction → Alert

### Key Features
- **Synthetic data generator** (no real banking data needed)
- **Multiple ML models** (Logistic Regression, Random Forest, XGBoost)
- **Handling imbalance** with SMOTE & class weights
- **Comprehensive evaluation** (ROC-AUC, PR-AUC, confusion matrix)
- **Threshold optimization** for business costs
- **Feature importance** analysis

## 📊 Dataset

Synthetic transactions with:
- **50,000 transactions** (default) - configurable
- **0.5% fraud rate** (realistic imbalance)
- **Features**: amount, time, location, merchant, device type, customer history
- **Engineered features**: log_amount, velocity ratios, time-based patterns

## 🛠 Tech Stack

| Category | Tools |
|----------|-------|
| Data Processing | pandas, numpy, scikit-learn |
| ML Models | Logistic Regression, Random Forest, XGBoost |
| Imbalance Handling | SMOTE, class_weight='balanced' |
| Visualization | matplotlib, seaborn |
| Evaluation | precision, recall, F1, ROC-AUC, PR-AUC |

## 📁 Project Structure
```
Credit-Card-Fraud-Detection/
│
├── data/ # Generated synthetic data
├── src/ # Source code modules
│ ├── data_generator.py
│ ├── preprocess.py
│ ├── train.py
│ ├── evaluate.py
│ └── visualize.py
├── models/ # Saved trained models
├── outputs/ # Plots and reports
├── main.py # Run complete pipeline
├── requirements.txt # Dependencies
└── README.md
```

📈 Results

Model Performance
Model	ROC-AUC	PR-AUC	Precision	Recall	F1-Score
XGBoost	0.95+	0.75+	0.85+	0.72+	0.78+
Random Forest	0.94+	0.73+	0.83+	0.70+	0.76+
Logistic Regression	0.89+	0.65+	0.75+	0.62+	0.68+


Key Insights
Top predictors: amount, transaction hour, velocity features

Fraud patterns: Higher at night, larger amounts, international

Optimal threshold: ~0.3-0.4 balances precision/recall

📊 Visualizations
The system generates:

Class distribution - Shows fraud imbalance

Amount distributions - Normal vs fraud patterns

Hourly patterns - Time-based fraud analysis

Correlation matrix - Feature relationships

ROC & PR curves - Model performance

Confusion matrix - Prediction breakdown

Feature importance - Top fraud indicators

💡 Interpretation Guide

Metric	Good Value	Why It Matters

Recall	> 0.70	Catching fraud (FN cost high)

Precision	> 0.50	Avoiding false alerts (FP cost)

PR-AUC	> 0.70	Better for imbalanced data

Optimal Threshold	0.3-0.5	Business cost dependent