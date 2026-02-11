# Circles Binary Classification
Author:Sneha pandey

### Assignment Overview
This repository contains the solution for the **Binary Classification with Neural Networks (Circles Dataset)** assignment.

 Objective: 
The goal is to build, train, evaluate, and compare different PyTorch artificial neural network (ANN) architectures to classify a synthetic circular dataset.

---

## Dataset
- `circles_binary_classification.csv`:  
  Contains two input features `X1` and `X2` and a binary label `label`.
- Format: CSV file with columns:
  - `X1`, `X2`: feature coordinates  
  - `label`: binary target (0 or 1)

---

## Notebook
- `circles_binary_classification.ipynb`:
  - Loads and inspects the dataset
  - Visualizes the data with scatter plots
  - Performs train-test split
  - Implements three PyTorch models:
    - **Model V0:** 2 → 5 → 1 (linear)
    - **Model V1:** 2 → 15 → 15 → 1 (linear)
    - **Model V2:** 2 → 64 → 64 → 10 → 1 with ReLU (nonlinear)
  - Trains models using **BCEWithLogitsLoss**
  - Evaluates models with accuracy metric and decision boundary plots
  - Compares **SGD vs Adam** optimizer (optional extra credit)
  - Plots training and testing loss curves
  - Includes untrained vs trained predictions

---

## Key Features
- Device-agnostic code (runs on CPU or GPU)
- Reproducible results (`torch.manual_seed(42)`)
- Clear visualization of model performance
- Reinitialization of models and optimizers before each experiment for fair comparison
- Step-by-step workflow aligned with assignment instructions..

