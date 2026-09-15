# Predictive Maintenance Using Machine Learning

A machine learning project for predicting the **Remaining Useful Life (RUL)** of aircraft engines using sensor and operational data.

The project uses the **NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset** and focuses initially on the **FD001 subset**.

## Project Objective

The goal of this project is to estimate how many operational cycles an engine has remaining before failure.

Predicting RUL can support predictive maintenance by helping identify engines that may require maintenance before failure occurs.

## Dataset

This project uses the NASA C-MAPSS dataset.

The dataset contains:

- Engine identifiers
- Operational cycles
- Operational settings
- Multiple sensor measurements
- RUL values for training data
- Separate RUL ground-truth values for the test data

The C-MAPSS dataset contains four subsets:

- FD001
- FD002
- FD003
- FD004

**Current project status:** FD001 has been completed. The remaining subsets are planned for future work.

## Project Workflow

```text
Raw C-MAPSS Data
        ↓
Data Exploration
        ↓
Data Cleaning
        ↓
RUL Calculation
        ↓
RUL Capping
        ↓
Feature Engineering
        ↓
Rolling Mean & Standard Deviation
        ↓
Model Training
        ↓
Test Prediction
        ↓
Model Evaluation
