#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import json
from datetime import datetime
from sklearn.metrics import average_precision_score

from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# Create output directories for results
def create_directories():
    """Create directories for saving model and plots if they don't exist."""
    os.makedirs('../../models_retrained', exist_ok=True)
    os.makedirs('plots_2', exist_ok=True)


def load_data(data_path):
    """
    Load the dataset.

    Args:
        data_path: Path to the dataset

    Returns:
        X: Features
        y: Target variable
    """
    print(f"Loading data from {data_path}")
    try:
        # Load the CSV file
        data = pd.read_csv(data_path)

        # Check if the dataset has an unnamed index column
        if data.columns[0] == 'Unnamed: 0' or data.columns[0] == '':
            print("Detected unnamed index column, setting it as index")
            data.set_index(data.columns[0], inplace=True)

        # Check for target variable - assume it's the first column
        # This is just a default; you should modify this according to your data structure
        if len(data.columns) == 0:
            raise ValueError("No columns found in the dataset after processing")

        # Print column names to help with debugging
        print("Column names in dataset:")
        for i, col in enumerate(data.columns):
            print(f"  {i}: {col}")

        # Split features and target
        y = data['Car'] # First column as target
        X = data.drop(columns=['Car'])  # All columns except the first one

        print(f"Data loaded successfully: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Target variable type: {y.dtype}")
        print(f"Target variable unique values: {y.unique()}")

        return X, y
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please check your data format and structure")
        raise


def split_data(X, y, test_size=0.2, valid_size=0.25):
    """
    Split data into train, validation, and test sets.
    Use stratified splitting to maintain class distribution.

    Args:
        X: Features
        y: Target variable
        test_size: Proportion of data for test set
        valid_size: Proportion of non-test data for validation set

    Returns:
        X_train, X_valid, X_test, y_train, y_valid, y_test
    """
    # Check if y values are numeric for stratification
    if isinstance(y, pd.Series) and y.dtype == object:
        print("Warning: Target variable contains non-numeric values. Converting to numeric for stratification.")
        # Use label encoder to convert to numeric for stratification purposes
        temp_encoder = LabelEncoder()
        stratify_y = temp_encoder.fit_transform(y)
    else:
        stratify_y = y

    # First split to separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=stratify_y
    )

    # For second split, check again
    if isinstance(y_temp, pd.Series) and y_temp.dtype == object:
        stratify_temp = temp_encoder.transform(y_temp)
    else:
        stratify_temp = y_temp

    # Second split to get training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp, y_temp, test_size=valid_size, random_state=RANDOM_SEED, stratify=stratify_temp
    )

    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_valid.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Print class distribution
    try:
        # Convert to numpy array if needed
        if isinstance(y_train, pd.Series):
            y_train_array = y_train.values
            y_valid_array = y_valid.values
            y_test_array = y_test.values
        else:
            y_train_array = y_train
            y_valid_array = y_valid
            y_test_array = y_test

        print("\nClass distribution:")
        print(f"Train set: {np.bincount(y_train_array.astype(int))}")
        print(f"Validation set: {np.bincount(y_valid_array.astype(int))}")
        print(f"Test set: {np.bincount(y_test_array.astype(int))}")
    except:
        print("\nCould not print class distribution. Target variable may not be suitable for bincount.")

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def preprocess_data(X_train, X_valid, X_test, y_train, y_valid, y_test):
    """
    Preprocess the data.
    In this case, we're standardizing the features and encoding labels if needed.

    Args:
        X_train, X_valid, X_test: Feature sets
        y_train, y_valid, y_test: Target variables

    Returns:
        X_train_scaled, X_valid_scaled, X_test_scaled: Scaled feature sets
        y_train_encoded, y_valid_encoded, y_test_encoded: Encoded target variables
        scaler: Fitted StandardScaler
        label_encoder: Fitted LabelEncoder (if needed)
    """
    # Feature scaling - handle both numpy arrays and pandas DataFrames
    scaler = StandardScaler()

    if isinstance(X_train, pd.DataFrame):
        # Keep a copy of the original column names
        feature_names = X_train.columns

        # Fit and transform on the training data
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            index=X_train.index,
            columns=feature_names
        )

        # Transform validation and test sets
        X_valid_scaled = pd.DataFrame(
            scaler.transform(X_valid),
            index=X_valid.index,
            columns=feature_names
        )

        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            index=X_test.index,
            columns=feature_names
        )
    else:
        # If numpy arrays, just apply the transforms directly
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)
        X_test_scaled = scaler.transform(X_test)

    # Check if label encoding is needed
    # Handle both pandas Series and numpy arrays
    if (isinstance(y_train, pd.Series) and y_train.dtype == object) or \
            (not isinstance(y_train, pd.Series) and isinstance(y_train[0], str)):
        print("Encoding non-numeric labels...")
        label_encoder = LabelEncoder()

        if isinstance(y_train, pd.Series):
            y_train_encoded = pd.Series(
                label_encoder.fit_transform(y_train),
                index=y_train.index
            )
            y_valid_encoded = pd.Series(
                label_encoder.transform(y_valid),
                index=y_valid.index
            )
            y_test_encoded = pd.Series(
                label_encoder.transform(y_test),
                index=y_test.index
            )
        else:
            y_train_encoded = label_encoder.fit_transform(y_train)
            y_valid_encoded = label_encoder.transform(y_valid)
            y_test_encoded = label_encoder.transform(y_test)

        print(f"Classes mapped as: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
        return X_train_scaled, X_valid_scaled, X_test_scaled, y_train_encoded, y_valid_encoded, y_test_encoded, scaler, label_encoder
    else:
        # If labels are already numeric, no encoding needed
        print("Target variable is already numeric. No encoding needed.")
        return X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test, scaler, None


def train_model(X_train, y_train, params):
    """
    Train an XGBoost binary classification model.

    Args:
        X_train: Training features
        y_train: Training target
        params: Model parameters

    Returns:
        Trained model
    """
    # Create XGBoost classifier with binary:logistic objective
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        colsample_bytree=params['model__colsample_bytree'],
        learning_rate=params['model__learning_rate'],
        max_depth=params['model__max_depth'],
        min_child_weight=params['model__min_child_weight'],
        n_estimators=params['model__n_estimators'],
        subsample=params['model__subsample'],
        random_state=RANDOM_SEED,
        use_label_encoder=False,  # Avoid warning
        verbosity=1
    )

    print("Training XGBoost Binary Classification model...")
    model.fit(X_train, y_train, verbose=False)
    print("Model training completed.")

    return model


def perform_cross_validation(X_train, y_train, params, n_folds=5):
    """
    Perform cross-validation on the training set for binary classification.

    Args:
        X_train: Training features
        y_train: Training target
        params: Model parameters
        n_folds: Number of folds for cross-validation

    Returns:
        cv_scores: Cross-validation scores
        cv_predictions: Cross-validation predictions
        cv_probabilities: Cross-validation prediction probabilities
    """
    # Convert y_train to numpy array if it's a pandas Series or DataFrame
    if isinstance(y_train, pd.Series) or isinstance(y_train, pd.DataFrame):
        y_train_values = y_train.values
    else:
        y_train_values = y_train

    # Use stratified k-fold to maintain class balance
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = []

    # Initialize arrays for predictions
    if isinstance(y_train, pd.Series):
        cv_predictions = np.zeros(len(y_train))
        cv_probabilities = np.zeros(len(y_train))  # For binary classification
    else:
        cv_predictions = np.zeros_like(y_train_values)
        cv_probabilities = np.zeros_like(y_train_values, dtype=float)

    print(f"Performing {n_folds}-fold cross-validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train_values), 1):
        # Handle different types of X_train (numpy array or DataFrame)
        if isinstance(X_train, np.ndarray):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        else:  # Assuming DataFrame or similar
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]

        # Handle different types of y_train
        if isinstance(y_train, pd.Series) or isinstance(y_train, pd.DataFrame):
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        else:
            y_fold_train, y_fold_val = y_train_values[train_idx], y_train_values[val_idx]

        # Train model - specifically binary classification
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            colsample_bytree=params['model__colsample_bytree'],
            learning_rate=params['model__learning_rate'],
            max_depth=params['model__max_depth'],
            min_child_weight=params['model__min_child_weight'],
            n_estimators=params['model__n_estimators'],
            subsample=params['model__subsample'],
            random_state=RANDOM_SEED,
            use_label_encoder=False,
            verbosity=0
        )
        model.fit(X_fold_train, y_fold_train, verbose=False)

        # Predict and evaluate
        y_pred = model.predict(X_fold_val)
        y_prob = model.predict_proba(X_fold_val)[:, 1]  # Probability of class 1

        # Store predictions and probabilities
        if isinstance(X_train, np.ndarray):
            cv_predictions[val_idx] = y_pred
            cv_probabilities[val_idx] = y_prob
        else:
            # For DataFrame, we need to match the indices
            for i, idx in enumerate(val_idx):
                cv_predictions[idx] = y_pred[i]
                cv_probabilities[idx] = y_prob[i]

        # Calculate binary classification metrics
        accuracy = accuracy_score(y_fold_val, y_pred)
        precision = precision_score(y_fold_val, y_pred, average='binary')
        recall = recall_score(y_fold_val, y_pred, average='binary')
        f1 = f1_score(y_fold_val, y_pred, average='binary')

        cv_scores.append({
            'fold': fold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

        print(
            f"Fold {fold}: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}")

    # Calculate average CV scores
    avg_accuracy = np.mean([score['accuracy'] for score in cv_scores])
    avg_precision = np.mean([score['precision'] for score in cv_scores])
    avg_recall = np.mean([score['recall'] for score in cv_scores])
    avg_f1 = np.mean([score['f1'] for score in cv_scores])

    print(f"Cross-validation results (average):")
    print(f"Accuracy: {avg_accuracy:.4f}")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall: {avg_recall:.4f}")
    print(f"F1 Score: {avg_f1:.4f}")

    return cv_scores, cv_predictions, cv_probabilities, y_train

    # Train model
    model = xgb.XGBClassifier(
        objective=objective,
        colsample_bytree=params['model__colsample_bytree'],
        learning_rate=params['model__learning_rate'],
        max_depth=params['model__max_depth'],
        min_child_weight=params['model__min_child_weight'],
        n_estimators=params['model__n_estimators'],
        subsample=params['model__subsample'],
        random_state=RANDOM_SEED,
        use_label_encoder=False,
        verbosity=0
    )
    model.fit(X_fold_train, y_fold_train, verbose=False)

    # Predict and evaluate
    y_pred = model.predict(X_fold_val)
    y_prob = model.predict_proba(X_fold_val)

    # Store predictions and probabilities
    cv_predictions[val_idx] = y_pred
    if is_binary:
        cv_probabilities[val_idx] = y_prob[:, 1]  # Probability of positive class
    else:
        for i, idx in enumerate(val_idx):
            cv_probabilities[idx] = y_prob[i]

    # Calculate metrics
    accuracy = accuracy_score(y_fold_val, y_pred)

    if is_binary:
        precision = precision_score(y_fold_val, y_pred, average='binary')
        recall = recall_score(y_fold_val, y_pred, average='binary')
        f1 = f1_score(y_fold_val, y_pred, average='binary')
    else:
        precision = precision_score(y_fold_val, y_pred, average='weighted')
        recall = recall_score(y_fold_val, y_pred, average='weighted')
        f1 = f1_score(y_fold_val, y_pred, average='weighted')

    cv_scores.append({
        'fold': fold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })

    print(f"Fold {fold}: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}")

def evaluate_model(model, X, y, dataset_name):
    """
    Evaluate the model on a given dataset for binary classification.

    Args:
        model: Trained model
        X: Features
        y: Target variable
        dataset_name: Name of the dataset (train, valid, test)

    Returns:
        Dictionary with evaluation metrics
    """
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]  # Probability of positive class (class 1)

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='binary')
    recall = recall_score(y, y_pred, average='binary')
    f1 = f1_score(y, y_pred, average='binary')
    conf_matrix = confusion_matrix(y, y_pred)

    print(f"{dataset_name} set evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")

    # Calculate and print additional binary classification metrics
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"True Positives: {tp}, False Positives: {fp}")
    print(f"True Negatives: {tn}, False Negatives: {fn}")
    print(f"Specificity (True Negative Rate): {specificity:.4f}")

    # Print classification report
    class_report = classification_report(y, y_pred)
    print(f"Classification Report:\n{class_report}")

    return {
        'y_true': y,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'conf_matrix': conf_matrix,
        'class_report': class_report,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp
    }


def plot_results(train_results, valid_results, test_results, cv_results):
    """
    Create plots for binary classification model evaluation.

    Args:
        train_results: Training set evaluation results
        valid_results: Validation set evaluation results
        test_results: Test set evaluation results
        cv_results: Cross-validation results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Plot 1: Confusion matrices for all datasets
    plt.figure(figsize=(18, 6))

    # Training set confusion matrix
    plt.subplot(1, 3, 1)
    cm = train_results['conf_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Training Set: Confusion Matrix\nAccuracy = {train_results["accuracy"]:.4f}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Validation set confusion matrix
    plt.subplot(1, 3, 2)
    cm = valid_results['conf_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Validation Set: Confusion Matrix\nAccuracy = {valid_results["accuracy"]:.4f}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Test set confusion matrix
    plt.subplot(1, 3, 3)
    cm = test_results['conf_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Test Set: Confusion Matrix\nAccuracy = {test_results["accuracy"]:.4f}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.tight_layout()
    plt.savefig(f'plots_2/confusion_matrices_{timestamp}.png')

    # Plot 2: ROC curves
    plt.figure(figsize=(15, 5))

    # Training set ROC curve
    plt.subplot(1, 3, 1)
    fpr_train, tpr_train, _ = roc_curve(train_results['y_true'], train_results['y_prob'])
    roc_auc_train = auc(fpr_train, tpr_train)
    plt.plot(fpr_train, tpr_train, lw=2, label=f'ROC curve (AUC = {roc_auc_train:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Training Set: ROC Curve')
    plt.legend(loc="lower right")

    # Validation set ROC curve
    plt.subplot(1, 3, 2)
    fpr_valid, tpr_valid, _ = roc_curve(valid_results['y_true'], valid_results['y_prob'])
    roc_auc_valid = auc(fpr_valid, tpr_valid)
    plt.plot(fpr_valid, tpr_valid, lw=2, label=f'ROC curve (AUC = {roc_auc_valid:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Validation Set: ROC Curve')
    plt.legend(loc="lower right")

    # Test set ROC curve
    plt.subplot(1, 3, 3)
    fpr_test, tpr_test, _ = roc_curve(test_results['y_true'], test_results['y_prob'])
    roc_auc_test = auc(fpr_test, tpr_test)
    plt.plot(fpr_test, tpr_test, lw=2, label=f'ROC curve (AUC = {roc_auc_test:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test Set: ROC Curve')
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(f'plots_2/roc_curves_{timestamp}.png')

    # Plot 3: Precision-Recall curves
    plt.figure(figsize=(15, 5))

    # Training set Precision-Recall curve
    plt.subplot(1, 3, 1)
    precision_train, recall_train, _ = precision_recall_curve(train_results['y_true'], train_results['y_prob'])
    avg_precision_train = average_precision_score(train_results['y_true'], train_results['y_prob'])
    plt.plot(recall_train, precision_train, lw=2, label=f'AP = {avg_precision_train:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Training Set: Precision-Recall Curve')
    plt.legend(loc="lower left")

    # Validation set Precision-Recall curve
    plt.subplot(1, 3, 2)
    precision_valid, recall_valid, _ = precision_recall_curve(valid_results['y_true'], valid_results['y_prob'])
    avg_precision_valid = average_precision_score(valid_results['y_true'], valid_results['y_prob'])
    plt.plot(recall_valid, precision_valid, lw=2, label=f'AP = {avg_precision_valid:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Validation Set: Precision-Recall Curve')
    plt.legend(loc="lower left")

    # Test set Precision-Recall curve
    plt.subplot(1, 3, 3)
    precision_test, recall_test, _ = precision_recall_curve(test_results['y_true'], test_results['y_prob'])
    avg_precision_test = average_precision_score(test_results['y_true'], test_results['y_prob'])
    plt.plot(recall_test, precision_test, lw=2, label=f'AP = {avg_precision_test:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Test Set: Precision-Recall Curve')
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(f'plots_2/precision_recall_curves_{timestamp}.png')

    # Plot 4: Cross-validation results
    plt.figure(figsize=(12, 6))
    folds = [score['fold'] for score in cv_results[0]]
    accuracy_scores = [score['accuracy'] for score in cv_results[0]]
    f1_scores = [score['f1'] for score in cv_results[0]]
    precision_scores = [score['precision'] for score in cv_results[0]]
    recall_scores = [score['recall'] for score in cv_results[0]]

    plt.plot(folds, accuracy_scores, 'o-', label='Accuracy')
    plt.plot(folds, f1_scores, 's-', label='F1 Score')
    plt.plot(folds, precision_scores, '^-', label='Precision')
    plt.plot(folds, recall_scores, 'D-', label='Recall')

    plt.title('Cross-Validation: Metrics by Fold')
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.xticks(folds)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'plots_2/cross_validation_{timestamp}.png')

    # Plot 5: Probability distributions by class
    plt.figure(figsize=(15, 5))

    # Training set probability distribution
    plt.subplot(1, 3, 1)
    probas_0 = [p for i, p in enumerate(train_results['y_prob']) if train_results['y_true'].iloc[i] == 0] if isinstance(
        train_results['y_true'], pd.Series) else [p for i, p in enumerate(train_results['y_prob']) if
                                                  train_results['y_true'][i] == 0]
    probas_1 = [p for i, p in enumerate(train_results['y_prob']) if train_results['y_true'].iloc[i] == 1] if isinstance(
        train_results['y_true'], pd.Series) else [p for i, p in enumerate(train_results['y_prob']) if
                                                  train_results['y_true'][i] == 1]

    plt.hist(probas_0, bins=20, alpha=0.5, color='blue', label='Class 0')
    plt.hist(probas_1, bins=20, alpha=0.5, color='red', label='Class 1')
    plt.title('Training Set: Probability Distribution by Class')
    plt.xlabel('Predicted Probability for Class 1')
    plt.ylabel('Frequency')
    plt.legend()

    # Validation set probability distribution
    plt.subplot(1, 3, 2)
    probas_0 = [p for i, p in enumerate(valid_results['y_prob']) if valid_results['y_true'].iloc[i] == 0] if isinstance(
        valid_results['y_true'], pd.Series) else [p for i, p in enumerate(valid_results['y_prob']) if
                                                  valid_results['y_true'][i] == 0]
    probas_1 = [p for i, p in enumerate(valid_results['y_prob']) if valid_results['y_true'].iloc[i] == 1] if isinstance(
        valid_results['y_true'], pd.Series) else [p for i, p in enumerate(valid_results['y_prob']) if
                                                  valid_results['y_true'][i] == 1]

    plt.hist(probas_0, bins=20, alpha=0.5, color='blue', label='Class 0')
    plt.hist(probas_1, bins=20, alpha=0.5, color='red', label='Class 1')
    plt.title('Validation Set: Probability Distribution by Class')
    plt.xlabel('Predicted Probability for Class 1')
    plt.ylabel('Frequency')
    plt.legend()

    # Test set probability distribution
    plt.subplot(1, 3, 3)
    probas_0 = [p for i, p in enumerate(test_results['y_prob']) if test_results['y_true'].iloc[i] == 0] if isinstance(
        test_results['y_true'], pd.Series) else [p for i, p in enumerate(test_results['y_prob']) if
                                                 test_results['y_true'][i] == 0]
    probas_1 = [p for i, p in enumerate(test_results['y_prob']) if test_results['y_true'].iloc[i] == 1] if isinstance(
        test_results['y_true'], pd.Series) else [p for i, p in enumerate(test_results['y_prob']) if
                                                 test_results['y_true'][i] == 1]

    plt.hist(probas_0, bins=20, alpha=0.5, color='blue', label='Class 0')
    plt.hist(probas_1, bins=20, alpha=0.5, color='red', label='Class 1')
    plt.title('Test Set: Probability Distribution by Class')
    plt.xlabel('Predicted Probability for Class 1')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'plots_2/probability_distribution_{timestamp}.png')

    print(f"All plots saved in 'plots_2/' directory with timestamp {timestamp}")

    # Plot 7: Decision boundaries or thresholds
    try:
        # Create a plot showing different performance metrics at different threshold values
        thresholds = np.linspace(0, 1, 100)
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []

        for threshold in thresholds:
            y_pred = (test_results['y_prob'] >= threshold).astype(int)
            accuracies.append(accuracy_score(test_results['y_true'], y_pred))

            # Handle division by zero warnings
            with np.errstate(divide='ignore', invalid='ignore'):
                precisions.append(precision_score(test_results['y_true'], y_pred, zero_division=0))
                recalls.append(recall_score(test_results['y_true'], y_pred, zero_division=0))
                f1_scores.append(f1_score(test_results['y_true'], y_pred, zero_division=0))

        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, accuracies, label='Accuracy')
        plt.plot(thresholds, precisions, label='Precision')
        plt.plot(thresholds, recalls, label='Recall')
        plt.plot(thresholds, f1_scores, label='F1 Score')

        # Add vertical line at default threshold (0.5)
        plt.axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label='Default threshold (0.5)')

        plt.title('Performance Metrics vs. Classification Threshold')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'plots_2/threshold_performance_{timestamp}.png')
    except:
        print("Could not create threshold performance plot.")

    print(f"All plots saved in 'plots_2/' directory with timestamp {timestamp}")

    plt.tight_layout()
    plt.savefig(f'plots_2/confusion_matrices_{timestamp}.png')

    # Plot 4: Cross-validation results
    plt.figure(figsize=(12, 6))
    folds = [score['fold'] for score in cv_results[0]]
    accuracy_scores = [score['accuracy'] for score in cv_results[0]]
    f1_scores = [score['f1'] for score in cv_results[0]]
    precision_scores = [score['precision'] for score in cv_results[0]]
    recall_scores = [score['recall'] for score in cv_results[0]]

    plt.plot(folds, accuracy_scores, 'o-', label='Accuracy')
    plt.plot(folds, f1_scores, 's-', label='F1 Score')
    plt.plot(folds, precision_scores, '^-', label='Precision')
    plt.plot(folds, recall_scores, 'D-', label='Recall')

    plt.title('Cross-Validation: Metrics by Fold')
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.xticks(folds)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'plots_2/cross_validation_{timestamp}.png')

    print(f"All plots saved in 'plots/' directory with timestamp {timestamp}")


def save_model(model, scaler, label_encoder, metrics, params):
    """
    Save the trained model, scaler, metrics, and parameters.

    Args:
        model: Trained model
        scaler: Fitted scaler
        label_encoder: Fitted label encoder (if used)
        metrics: Evaluation metrics
        params: Model parameters
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_file = f'../../models_retrained/xgboost_binary_classifier_{timestamp}.joblib'

    # Create a dictionary with all components to save
    save_dict = {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'params': params,
        'metrics': metrics,
        'timestamp': timestamp
    }

    joblib.dump(save_dict, model_file)
    print(f"Model and associated data saved to {model_file}")
    # Create a simple model card with key metrics
    model_card = {
        'model_name': 'XGBoost Binary Classifier',
        'created_at': timestamp,
        'parameters': params,
        'performance': {
            'train': {
                'accuracy': metrics['train']['accuracy'],
                'precision': metrics['train']['precision'],
                'recall': metrics['train']['recall'],
                'f1': metrics['train']['f1'],
                'specificity': metrics['train']['specificity']
            },
            'validation': {
                'accuracy': metrics['valid']['accuracy'],
                'precision': metrics['valid']['precision'],
                'recall': metrics['valid']['recall'],
                'f1': metrics['valid']['f1'],
                'specificity': metrics['valid']['specificity']
            },
            'test': {
                'accuracy': metrics['test']['accuracy'],
                'precision': metrics['test']['precision'],
                'recall': metrics['test']['recall'],
                'f1': metrics['test']['f1'],
                'specificity': metrics['test']['specificity']
            },
            'cross_validation': {
                'avg_accuracy': metrics['cv']['avg_accuracy'],
                'avg_precision': metrics['cv']['avg_precision'],
                'avg_recall': metrics['cv']['avg_recall'],
                'avg_f1': metrics['cv']['avg_f1']
            }
        }
    }

    # Save the model card
    card_file = f'../../models_retrained/model_card_{timestamp}.json'
    with open(card_file, 'w') as f:
        json.dump(model_card, f, indent=4)
    print(f"Model card saved to {card_file}")


def main(data_path, model_params):
    """
    Main function to run the entire workflow.

    Args:
        data_path: Path to the dataset
        model_params: Model parameters
    """
    # Create necessary directories
    create_directories()

    # Load data
    X, y = load_data(data_path)

    # Split data
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(X, y)

    # Preprocess data
    processed_data = preprocess_data(X_train, X_valid, X_test, y_train, y_valid, y_test)
    X_train_scaled, X_valid_scaled, X_test_scaled = processed_data[0:3]
    y_train_encoded, y_valid_encoded, y_test_encoded = processed_data[3:6]
    scaler, label_encoder = processed_data[6:8]

    # Perform cross-validation
    cv_results = perform_cross_validation(X_train_scaled, y_train_encoded, model_params)

    # Train final model
    model = train_model(X_train_scaled, y_train_encoded, model_params)

    # Evaluate model on all datasets
    train_results = evaluate_model(model, X_train_scaled, y_train_encoded, "Training")
    valid_results = evaluate_model(model, X_valid_scaled, y_valid_encoded, "Validation")
    test_results = evaluate_model(model, X_test_scaled, y_test_encoded, "Test")

    # Plot results
    plot_results(train_results, valid_results, test_results, cv_results)

    # Save model and associated data
    metrics = {
        'train': {k: v for k, v in train_results.items() if k not in ['y_true', 'y_pred', 'y_prob']},
        'valid': {k: v for k, v in valid_results.items() if k not in ['y_true', 'y_pred', 'y_prob']},
        'test': {k: v for k, v in test_results.items() if k not in ['y_true', 'y_pred', 'y_prob']},
        'cv': {
            'scores': cv_results[0],
            'avg_accuracy': np.mean([score['accuracy'] for score in cv_results[0]]),
            'avg_precision': np.mean([score['precision'] for score in cv_results[0]]),
            'avg_recall': np.mean([score['recall'] for score in cv_results[0]]),
            'avg_f1': np.mean([score['f1'] for score in cv_results[0]])
        }
    }
    save_model(model, scaler, label_encoder, metrics, model_params)


if __name__ == "__main__":
    # Define model parameters
    model_params = {
        'model__colsample_bytree': 1.0,
        'model__learning_rate': 0.1,
        'model__max_depth': 5,
        'model__min_child_weight': 5,
        'model__n_estimators': 300,
        'model__subsample': 0.8
    }

    # Path to your dataset
    data_path = "class_data.csv"  # Replace with your data path

    # Run the main workflow
    main(data_path, model_params)