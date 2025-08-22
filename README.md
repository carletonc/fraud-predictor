# ML Fraud Predictor Monitoring  

🎯 **A production-ready machine learning application demonstrating ML engineering best practices**  

## 🌟 About  

This repository contains a simple dashboard for demonstrating **online model monitoring** and **feature drift detection**. It assumes the presence of an online feedback mechanism to validate predictions, enabling monitoring of precision, recall, and feature drift for the most influential features.  

### Accuracy Monitoring  

Our offline “gold” test set is benchmarked against a probability threshold of **0.5**.  
In production, the threshold is increased to **0.77** to prioritize precision at the cost of recall.  
This difference explains the noticeable performance gap between offline and online results.  
I may update the offline metrics in the future to align with the 0.77 production threshold.  

### Feature Drift Monitoring  

Feature drift is measured using **Jensen–Shannon Divergence (JSD)**, which compares normalized probability distributions between the offline test set and the online feature distributions.  
Currently, distributions are computed using **10 histogram bins**, though this may be expanded to capture more granular shifts.  

# ⚙️ Setup  

Clone the repository and install dependencies:  

```pip install -r requirements.txt```

Run the Streamlit app:

```streamlit run app.py```


## 📊 Model Training Methodology

The goal of this repo is to demonstrate batch online monitoring, not advanced modeling techniques.

The dataset was split into train, validation, and holdout test sets.

Hyperparameter tuning was performed via cross-validation and Bayesian optimization (10 iterations) strictly on the training set, optimizing for AUC-PR due to significant class imbalance.

After selecting the best parameters, the model was retrained on the training set with early stopping on the validation set.

Final performance was benchmarked on the holdout test set, which the model never saw during training.