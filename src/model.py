"""
model.py
--------
Reusable model training and evaluation functions for the
Financial Fraud Detection Pipeline.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score
)


def train_random_forest(X_train, y_train, n_estimators: int = 100,
                        max_depth: int = 12, random_state: int = 42):
    """
    Train a Random Forest classifier with balanced class weights.

    Args:
        X_train:      Training features
        y_train:      Training labels
        n_estimators: Number of trees
        max_depth:    Maximum tree depth
        random_state: Reproducibility seed

    Returns:
        Trained RandomForestClassifier
    """
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    print("Random Forest trained ✅")
    return rf


def train_xgboost(X_train, y_train, n_estimators: int = 200,
                  max_depth: int = 6, learning_rate: float = 0.05,
                  random_state: int = 42):
    """
    Train an XGBoost classifier with scale_pos_weight for imbalance.

    Args:
        X_train:       Training features
        y_train:       Training labels
        n_estimators:  Number of boosting rounds
        max_depth:     Maximum tree depth
        learning_rate: Step size shrinkage
        random_state:  Reproducibility seed

    Returns:
        Trained XGBClassifier
    """
    print("Training XGBoost...")
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=random_state,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    print("XGBoost trained ✅")
    return xgb


def evaluate_model(model, X_test, y_test, model_name: str = "Model"):
    """
    Print classification report and return metrics dictionary.

    Args:
        model:      Trained classifier
        X_test:     Test features
        y_test:     Test labels
        model_name: Display name for the model

    Returns:
        Dictionary of evaluation metrics
    """
    pred  = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    print("=" * 50)
    print(f"  {model_name}")
    print("=" * 50)
    print(classification_report(y_test, pred, target_names=['Legitimate', 'Fraud']))

    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    metrics = {
        'Model':                model_name,
        'Precision':            round(precision_score(y_test, pred), 4),
        'Recall':               round(recall_score(y_test, pred), 4),
        'F1':                   round(f1_score(y_test, pred), 4),
        'ROC-AUC':              round(roc_auc_score(y_test, proba), 4),
        'PR-AUC':               round(average_precision_score(y_test, proba), 4),
        'Fraud Caught (TP)':    int(tp),
        'Fraud Missed (FN)':    int(fn),
        'False Alarms (FP)':    int(fp),
    }
    return metrics


def plot_confusion_matrix(models: list, X_test, y_test, output_path: str = None):
    """
    Plot side-by-side confusion matrices for a list of (name, model) tuples.

    Args:
        models:      List of (name, trained_model) tuples
        X_test:      Test features
        y_test:      Test labels
        output_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 5))
    if len(models) == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models):
        pred = model.predict(X_test)
        cm = confusion_matrix(y_test, pred)
        sns.heatmap(cm, annot=True, fmt=',', cmap='Blues', ax=ax,
                    xticklabels=['Pred: Legit', 'Pred: Fraud'],
                    yticklabels=['Actual: Legit', 'Actual: Fraud'],
                    linewidths=0.5, cbar=False)
        tn, fp, fn, tp = cm.ravel()
        ax.set_title(f'{name}\nConfusion Matrix', fontsize=12, fontweight='bold')
        ax.set_xlabel(f'Fraud Caught: {tp} | Missed: {fn}', fontsize=9)

    plt.suptitle('Confusion Matrices — Test Set', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved to {output_path} ✅")
    plt.show()


def plot_roc_curves(models: list, X_test, y_test, output_path: str = None):
    """
    Plot ROC curves for a list of (name, model) tuples.

    Args:
        models:      List of (name, trained_model) tuples
        X_test:      Test features
        y_test:      Test labels
        output_path: Optional path to save the figure
    """
    colors = ['#4C72B0', '#DD8452', '#2ca02c', '#d62728']
    plt.figure(figsize=(8, 6))

    for (name, model), color in zip(models, colors):
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})', color=color, linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Baseline')
    plt.xlabel('False Positive Rate', fontsize=11)
    plt.ylabel('True Positive Rate', fontsize=11)
    plt.title('ROC Curve Comparison', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved to {output_path} ✅")
    plt.show()


def export_predictions(model, X_test, y_test, feature_cols,
                       model_name: str = 'Model', output_path: str = '../outputs/predictions.csv'):
    """
    Export test set with predictions, probabilities, and outcome labels to CSV for Tableau.

    Args:
        model:        Trained classifier
        X_test:       Test features
        y_test:       Test labels
        feature_cols: List of feature column names
        model_name:   Short name prefix for columns (e.g. 'XGB')
        output_path:  Path to save predictions CSV
    """
    pred  = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    export_df = pd.DataFrame(X_test, columns=feature_cols)
    export_df['Class']                          = y_test.values
    export_df[f'{model_name}_Prediction']       = pred
    export_df[f'{model_name}_Fraud_Prob']       = proba
    export_df[f'{model_name}_Label']            = pd.Series(pred).map({0: 'Legitimate', 1: 'Fraud'}).values
    export_df['Actual_Label']                   = pd.Series(y_test.values).map({0: 'Legitimate', 1: 'Fraud'}).values

    def outcome(row):
        actual = row['Class']
        predicted = row[f'{model_name}_Prediction']
        if actual == 1 and predicted == 1: return 'True Positive'
        if actual == 0 and predicted == 0: return 'True Negative'
        if actual == 0 and predicted == 1: return 'False Positive'
        if actual == 1 and predicted == 0: return 'False Negative'

    export_df[f'{model_name}_Outcome'] = export_df.apply(outcome, axis=1)
    export_df.to_csv(output_path, index=False)

    print(f"Exported {len(export_df):,} rows to {output_path} ✅")
    print(f"\nOutcome breakdown:")
    print(export_df[f'{model_name}_Outcome'].value_counts().to_string())


if __name__ == '__main__':
    # Quick smoke test
    train = pd.read_csv('../data/train_resampled.csv')
    test  = pd.read_csv('../data/test.csv')

    X_train = train.drop(columns=['Class'])
    y_train = train['Class']
    X_test  = test.drop(columns=['Class'])
    y_test  = test['Class']

    rf  = train_random_forest(X_train, y_train)
    xgb = train_xgboost(X_train, y_train)

    rf_metrics  = evaluate_model(rf,  X_test, y_test, 'Random Forest')
    xgb_metrics = evaluate_model(xgb, X_test, y_test, 'XGBoost')

    results = pd.DataFrame([rf_metrics, xgb_metrics]).set_index('Model')
    print("\n=== Model Comparison ===")
    print(results.to_string())

    plot_confusion_matrix([('Random Forest', rf), ('XGBoost', xgb)], X_test, y_test,
                          output_path='../outputs/confusion_matrices.png')
    plot_roc_curves([('Random Forest', rf), ('XGBoost', xgb)], X_test, y_test,
                    output_path='../outputs/roc_curves.png')
    export_predictions(xgb, X_test, y_test, X_test.columns.tolist(),
                       model_name='XGB', output_path='../outputs/predictions.csv')
