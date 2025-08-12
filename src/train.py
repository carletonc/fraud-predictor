import json
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, average_precision_score, classification_report
import matplotlib.pyplot as plt

from src.constants import SEED, TRAIN_DATA_PATH, SCALER_PATH, MODEL_PATH, MODEL_METRICS_PATH

# Static parameters (not tuned)
STATIC_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'aucpr',  # PR-AUC is better for imbalanced data
    'booster': 'gbtree',
    'verbosity': 0,
    'random_state': SEED,
    'early_stopping_rounds': 50,
}

def load_and_preprocess_data():
    """Load data and create train/val/test splits with scaling."""
    df = pd.read_csv(TRAIN_DATA_PATH).drop(columns=['Time'])
    print(df.shape)
    print(df['Class'].value_counts(normalize=True))
    
    y = df[['Class']]
    X = df.drop(columns=['Class'])

    # Three-way split: Train/Val/Test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=SEED, stratify=y_temp  # 0.25 of 0.8 = 0.2 overall
    )

    # Fit scaler once on training data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler





def optimize_hyperparameters(X_train_scaled, y_train):
    """Run Optuna hyperparameter optimization."""
    def objective(trial):
        """Optuna objective function for hyperparameter optimization."""
        # Hyperparameters to tune
        tuned_params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 100, 1000)
        }
        
        # Combine static and tuned parameters
        params = {**STATIC_PARAMS, **tuned_params}
        
        # 5-fold CV on training set only
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        cv_scores = []
        
        for train_idx, val_idx in cv.split(X_train_scaled, y_train):
            X_cv_train, X_cv_val = X_train_scaled.iloc[train_idx], X_train_scaled.iloc[val_idx]
            y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_cv_train, y_cv_train,
                eval_set=[(X_cv_val, y_cv_val)],
                verbose=False
            )
            
            y_pred = model.predict_proba(X_cv_val)[:, 1]
            # same calculation as `aucpr` from xgboost
            score = average_precision_score(y_cv_val, y_pred)
            cv_scores.append(score)
        
        return np.mean(cv_scores)
    
    print("Starting hyperparameter optimization...")
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=SEED))
    study.optimize(objective, n_trials=20, show_progress_bar=True)
    print(f"Best CV PR-AUC: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")
    return study


def train_final_model(X_train_scaled, X_val_scaled, y_train, y_val, best_params):
    """Train final model with best hyperparameters."""
    print("\nTraining final model...")
    final_model = xgb.XGBClassifier(**STATIC_PARAMS, **best_params)
    final_model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_train_scaled, y_train), (X_val_scaled, y_val)],
        verbose=False
    )
    return final_model


def evaluate_model(final_model, X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test):
    """Evaluate model and print metrics."""
    # Predictions and metrics
    y_test_pred = final_model.predict_proba(X_test_scaled)[:, 1]
    test_roc_auc = roc_auc_score(y_test, y_test_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_test_pred)
    test_pr_auc = auc(recall, precision)

    # Print classification reports
    S = 20
    print("="*S, "TRAIN METRICS", "="*S)
    print(classification_report(y_train, final_model.predict(X_train_scaled)))
    print("="*S, "VAL METRICS", "="*S)
    print(classification_report(y_val, final_model.predict(X_val_scaled)))
    print("="*S, "TEST METRICS", "="*S)
    print(classification_report(y_test, final_model.predict(X_test_scaled)))

    print(f"\n=== FINAL RESULTS ===")
    print(f"Test ROC-AUC: {test_roc_auc:.4f}")
    print(f"Test PR-AUC: {test_pr_auc:.4f}")
    
    return test_roc_auc, test_pr_auc, y_test_pred


def create_visualizations(final_model, X_train_scaled, X_test_scaled, y_test, y_test_pred, test_pr_auc):
    """Create and display model evaluation plots."""
    precision, recall, _ = precision_recall_curve(y_test, y_test_pred)
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 20))

    # Training/Validation curve
    results = final_model.evals_result()
    if 'validation_0' in results and 'aucpr' in results['validation_0']:
        epochs = len(results['validation_0']['aucpr'])
        axes[0].plot(range(epochs), results['validation_0']['aucpr'], label='Train' if 'validation_1' in results else 'Validation')
        if 'validation_1' in results and 'aucpr' in results['validation_1']:
            axes[0].plot(range(epochs), results['validation_1']['aucpr'], label='Validation')
        axes[0].set_title('PR-AUC During Training')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('PR-AUC')
        axes[0].legend()
        axes[0].grid(True)

    # Precision-Recall curve
    axes[1].plot(recall, precision, label=f'Test PR-AUC = {test_pr_auc:.4f}')
    axes[1].set_title('Precision-Recall Curve (Test Set)')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].legend()
    axes[1].grid(True)

    # Precision and Recall vs Threshold
    prec, rec, thresholds = precision_recall_curve(y_test, y_test_pred)
    axes[2].plot(thresholds, prec[:-1], label='Precision')
    axes[2].plot(thresholds, rec[:-1], label='Recall')
    axes[2].set_title('Precision and Recall vs Threshold')
    axes[2].set_xlabel('Threshold')
    axes[2].set_ylabel('Score')
    axes[2].legend()
    axes[2].grid(True)

    # Feature importance
    feature_importance = final_model.feature_importances_
    top_features = np.argsort(feature_importance)[-10:]
    axes[3].barh(range(len(top_features)), feature_importance[top_features])
    axes[3].set_yticks(range(len(top_features)))
    axes[3].set_yticklabels([X_train_scaled.columns[i] for i in top_features])
    axes[3].set_title('Top 10 Feature Importances')
    axes[3].set_xlabel('Importance')

    plt.tight_layout()
    plt.show()
    
    
def save_artifacts(final_model, scaler, y_test, X_test_scaled, test_roc_auc, test_pr_auc):
    """Save model, scaler, and metrics to disk."""
    # Save preprocessing and model
    joblib.dump(scaler, SCALER_PATH) 
    final_model.save_model(MODEL_PATH)

    # Prepare and save metrics
    test_metrics = classification_report(y_test, final_model.predict(X_test_scaled), output_dict=True)
    test_metrics['1'].update({
        'roc_auc': float(test_roc_auc),
        'pr_auc': float(test_pr_auc),
    })

    with open(MODEL_METRICS_PATH, 'w') as f:
        json.dump(test_metrics, f, indent=4)
    
    print("Model artifacts saved successfully!")


def main():
    """Main training pipeline."""
    # Load and preprocess data
    X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler = load_and_preprocess_data()
    
    # Optimize hyperparameters
    study = optimize_hyperparameters(X_train_scaled, y_train)
    
    # Train final model
    final_model = train_final_model(X_train_scaled, X_val_scaled, y_train, y_val, study.best_params)
    
    # Evaluate model
    test_roc_auc, test_pr_auc, y_test_pred = evaluate_model(
        final_model, X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    )
    
    # Create visualizations
    # create_visualizations(final_model, X_train_scaled, X_test_scaled, y_test, y_test_pred, test_pr_auc)
    
    # Save artifacts
    save_artifacts(final_model, scaler, y_test, X_test_scaled, test_roc_auc, test_pr_auc)

if __name__ == "__main__":
    main()