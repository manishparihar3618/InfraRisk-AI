# Methodology

## Data Collection

The project uses the Florida National Bridge Inventory dataset.

---

## Data Cleaning

- Removed unnecessary columns
- Handled missing values
- Converted categorical features
- Checked duplicate records

---

## Feature Engineering

Features were selected based on bridge characteristics including:

- Construction Year
- Average Daily Traffic
- Span Length
- Navigation
- Structural Information
- Scour Critical Rating
- Deck Structure Type

---

## Target Creation

Infrastructure Risk was created using bridge condition ratings.

Low Condition Score → High Risk

Medium Condition Score → Medium Risk

Good Condition Score → Low Risk

---

## Model Training

Three Machine Learning models were evaluated.

- Decision Tree
- Random Forest
- XGBoost

XGBoost achieved the best performance and was selected for deployment.
