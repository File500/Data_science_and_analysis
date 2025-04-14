#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# Create output directories for results
def create_directories():
    """Create directories for saving model and plots if they don't exist."""
    os.makedirs('../../models_retrained', exist_ok=True)
    os.makedirs('plots', exist_ok=True)


def load_data(data_path):
    """
    Load the dataset.

    Args:
        data_path: Path to the dataset

    Returns:
        X: Features
        y: Target variable
    """
    # This is a placeholder for your data loading logic
    # Replace this with your actual data loading code
    print(f"Loading data from {data_path}")
    try:
        # Assuming data is in CSV format - modify according to your data format
        data = pd.read_csv(data_path)

        # Assuming the last column is the target variable - modify as needed
        y = data['Price']
        X = data.drop(columns=['Price'])

        print(f"Data loaded successfully: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def split_data(X, y, test_size=0.2, valid_size=0.25):
    """
    Split data into train, validation, and test sets.

    Args:
        X: Features
        y: Target variable
        test_size: Proportion of data for test set
        valid_size: Proportion of non-test data for validation set

    Returns:
        X_train, X_valid, X_test, y_train, y_valid, y_test
    """
    # First split to separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED
    )

    # Second split to get training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp, y_temp, test_size=valid_size, random_state=RANDOM_SEED
    )

    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_valid.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def preprocess_data(X_train, X_valid, X_test):
    """
    Preprocess the data. In this case, we're standardizing the features.

    Args:
        X_train, X_valid, X_test: Feature sets

    Returns:
        X_train_scaled, X_valid_scaled, X_test_scaled: Scaled feature sets
        scaler: Fitted StandardScaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_valid_scaled, X_test_scaled, scaler


def train_model(X_train, y_train, params):
    """
    Train a Gradient Boosting model.

    Args:
        X_train: Training features
        y_train: Training target
        params: Model parameters

    Returns:
        Trained model
    """
    model = GradientBoostingRegressor(
        learning_rate=params['model__learning_rate'],
        max_depth=params['model__max_depth'],
        min_samples_leaf=params['model__min_samples_leaf'],
        min_samples_split=params['model__min_samples_split'],
        n_estimators=params['model__n_estimators'],
        random_state=RANDOM_SEED
    )

    print("Training Gradient Boosting Regression model...")
    model.fit(X_train, y_train)
    print("Model training completed.")

    return model


def perform_cross_validation(X_train, y_train, params, n_folds=5):
    """
    Perform cross-validation on the training set.

    Args:
        X_train: Training features
        y_train: Training target
        params: Model parameters
        n_folds: Number of folds for cross-validation

    Returns:
        cv_scores: Cross-validation scores
        cv_predictions: Cross-validation predictions
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = []
    cv_predictions = np.zeros_like(y_train)

    print(f"Performing {n_folds}-fold cross-validation...")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        # Split data
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Train model
        model = GradientBoostingRegressor(
            learning_rate=params['model__learning_rate'],
            max_depth=params['model__max_depth'],
            min_samples_leaf=params['model__min_samples_leaf'],
            min_samples_split=params['model__min_samples_split'],
            n_estimators=params['model__n_estimators'],
            random_state=RANDOM_SEED
        )
        model.fit(X_fold_train, y_fold_train)

        # Predict and evaluate
        y_pred = model.predict(X_fold_val)
        cv_predictions[val_idx] = y_pred
        mse = mean_squared_error(y_fold_val, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_fold_val, y_pred)

        cv_scores.append({
            'fold': fold,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        })

        print(f"Fold {fold}: RMSE = {rmse:.4f}, R² = {r2:.4f}")

    # Calculate average CV scores
    avg_mse = np.mean([score['mse'] for score in cv_scores])
    avg_rmse = np.mean([score['rmse'] for score in cv_scores])
    avg_r2 = np.mean([score['r2'] for score in cv_scores])

    print(f"Cross-validation results (average):")
    print(f"MSE: {avg_mse:.4f}")
    print(f"RMSE: {avg_rmse:.4f}")
    print(f"R²: {avg_r2:.4f}")

    return cv_scores, cv_predictions, y_train


def evaluate_model(model, X, y, dataset_name):
    """
    Evaluate the model on a given dataset.

    Args:
        model: Trained model
        X: Features
        y: Target variable
        dataset_name: Name of the dataset (train, valid, test)

    Returns:
        Dictionary with evaluation metrics
    """
    y_pred = model.predict(X)

    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"{dataset_name} set evaluation:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")

    return {
        'y_true': y,
        'y_pred': y_pred,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


def plot_results(train_results, valid_results, test_results, cv_results=None):
    """
    Create plots for model evaluation.

    Args:
        train_results: Training set evaluation results
        valid_results: Validation set evaluation results
        test_results: Test set evaluation results
        cv_results: Cross-validation results (optional)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Plot 1: Actual vs Predicted values for all datasets
    plt.figure(figsize=(18, 6))

    # Training set
    plt.subplot(1, 3, 1)
    plt.scatter(train_results['y_true'], train_results['y_pred'], alpha=0.5)
    plt.plot([min(train_results['y_true']), max(train_results['y_true'])],
             [min(train_results['y_true']), max(train_results['y_true'])],
             'r--')
    plt.title(f'Training Set: Actual vs Predicted\nR² = {train_results["r2"]:.4f}')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')

    # Validation set
    plt.subplot(1, 3, 2)
    plt.scatter(valid_results['y_true'], valid_results['y_pred'], alpha=0.5)
    plt.plot([min(valid_results['y_true']), max(valid_results['y_true'])],
             [min(valid_results['y_true']), max(valid_results['y_true'])],
             'r--')
    plt.title(f'Validation Set: Actual vs Predicted\nR² = {valid_results["r2"]:.4f}')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')

    # Test set
    plt.subplot(1, 3, 3)
    plt.scatter(test_results['y_true'], test_results['y_pred'], alpha=0.5)
    plt.plot([min(test_results['y_true']), max(test_results['y_true'])],
             [min(test_results['y_true']), max(test_results['y_true'])],
             'r--')
    plt.title(f'Test Set: Actual vs Predicted\nR² = {test_results["r2"]:.4f}')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')

    plt.tight_layout()
    plt.savefig(f'plots/actual_vs_predicted_{timestamp}.png')

    # Plot 2: Residuals for all datasets
    plt.figure(figsize=(18, 6))

    # Training set residuals
    plt.subplot(1, 3, 1)
    residuals_train = train_results['y_true'] - train_results['y_pred']
    plt.scatter(train_results['y_pred'], residuals_train, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'Training Set: Residuals\nRMSE = {train_results["rmse"]:.4f}')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')

    # Validation set residuals
    plt.subplot(1, 3, 2)
    residuals_valid = valid_results['y_true'] - valid_results['y_pred']
    plt.scatter(valid_results['y_pred'], residuals_valid, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'Validation Set: Residuals\nRMSE = {valid_results["rmse"]:.4f}')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')

    # Test set residuals
    plt.subplot(1, 3, 3)
    residuals_test = test_results['y_true'] - test_results['y_pred']
    plt.scatter(test_results['y_pred'], residuals_test, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'Test Set: Residuals\nRMSE = {test_results["rmse"]:.4f}')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')

    plt.tight_layout()
    plt.savefig(f'plots/residuals_{timestamp}.png')

    # Plot 3: Residuals histograms for all datasets
    plt.figure(figsize=(18, 6))

    # Training set residuals histogram
    plt.subplot(1, 3, 1)
    plt.hist(residuals_train, bins=30, alpha=0.7, color='blue')
    plt.title(f'Training Set: Residuals Distribution\nMAE = {train_results["mae"]:.4f}')
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')

    # Validation set residuals histogram
    plt.subplot(1, 3, 2)
    plt.hist(residuals_valid, bins=30, alpha=0.7, color='green')
    plt.title(f'Validation Set: Residuals Distribution\nMAE = {valid_results["mae"]:.4f}')
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')

    # Test set residuals histogram
    plt.subplot(1, 3, 3)
    plt.hist(residuals_test, bins=30, alpha=0.7, color='orange')
    plt.title(f'Test Set: Residuals Distribution\nMAE = {test_results["mae"]:.4f}')
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(f'plots/residuals_histogram_{timestamp}.png')

    # Plot 4: Cross-validation results (if available)
    if cv_results is not None:
        plt.figure(figsize=(15, 10))

        # CV predictions
        plt.subplot(2, 2, 1)
        y_true, y_pred = cv_results[2], cv_results[1]
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
        plt.title('Cross-Validation: Actual vs Predicted')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')

        # CV residuals
        plt.subplot(2, 2, 2)
        residuals_cv = y_true - y_pred
        plt.scatter(y_pred, residuals_cv, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Cross-Validation: Residuals')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')

        # CV residuals histogram
        plt.subplot(2, 2, 3)
        plt.hist(residuals_cv, bins=30, alpha=0.7, color='purple')
        plt.title('Cross-Validation: Residuals Distribution')
        plt.xlabel('Residual Value')
        plt.ylabel('Frequency')

        # CV scores across folds
        plt.subplot(2, 2, 4)
        folds = [score['fold'] for score in cv_results[0]]
        rmse_scores = [score['rmse'] for score in cv_results[0]]
        r2_scores = [score['r2'] for score in cv_results[0]]

        ax1 = plt.gca()
        ax1.bar(folds, rmse_scores, alpha=0.7, color='blue', label='RMSE')
        ax1.set_xlabel('Fold')
        ax1.set_ylabel('RMSE', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        ax2.plot(folds, r2_scores, 'r-o', label='R²')
        ax2.set_ylabel('R²', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        plt.title('Cross-Validation: Metrics by Fold')

        # Add a legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.tight_layout()
        plt.savefig(f'plots/cross_validation_{timestamp}.png')

    # Plot 5: Feature importance (if model provides it)
    try:
        plt.figure(figsize=(10, 8))
        feature_importance = model.feature_importances_

        # If we have feature names, use them
        if hasattr(X_train, 'columns'):
            feature_names = X_train.columns
        else:
            feature_names = [f'Feature {i}' for i in range(len(feature_importance))]

        # Sort features by importance
        indices = np.argsort(feature_importance)[::-1]
        sorted_feature_names = [feature_names[i] for i in indices]
        sorted_importance = feature_importance[indices]

        # Plot feature importance
        plt.barh(range(len(sorted_importance)), sorted_importance, align='center')
        plt.yticks(range(len(sorted_importance)), sorted_feature_names)
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(f'plots/feature_importance_{timestamp}.png')
    except (AttributeError, NameError):
        print("Could not create feature importance plot.")

    print(f"All plots saved in 'plots/' directory with timestamp {timestamp}")


def save_model(model, scaler, metrics, params):
    """
    Save the trained model, scaler, metrics, and parameters.

    Args:
        model: Trained model
        scaler: Fitted scaler
        metrics: Evaluation metrics
        params: Model parameters
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_file = f'../../models_retrained/gradient_boosting_model_{timestamp}.joblib'

    # Create a dictionary with all components to save
    save_dict = {
        'model': model,
        'scaler': scaler,
        'params': params,
        'metrics': metrics,
        'timestamp': timestamp
    }

    joblib.dump(save_dict, model_file)
    print(f"Model and associated data saved to {model_file}")


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
    X_train_scaled, X_valid_scaled, X_test_scaled, scaler = preprocess_data(X_train, X_valid, X_test)

    # Perform cross-validation
    cv_results = perform_cross_validation(X_train_scaled, y_train, model_params)

    # Train final model
    model = train_model(X_train_scaled, y_train, model_params)

    # Evaluate model on all datasets
    train_results = evaluate_model(model, X_train_scaled, y_train, "Training")
    valid_results = evaluate_model(model, X_valid_scaled, y_valid, "Validation")
    test_results = evaluate_model(model, X_test_scaled, y_test, "Test")

    # Plot results
    plot_results(train_results, valid_results, test_results, cv_results)

    # Save model and associated data
    metrics = {
        'train': train_results,
        'valid': valid_results,
        'test': test_results,
        'cv': {
            'scores': cv_results[0],
            'avg_mse': np.mean([score['mse'] for score in cv_results[0]]),
            'avg_rmse': np.mean([score['rmse'] for score in cv_results[0]]),
            'avg_r2': np.mean([score['r2'] for score in cv_results[0]])
        }
    }
    save_model(model, scaler, metrics, model_params)


if __name__ == "__main__":
    # Define model parameters
    model_params = {
        'model__learning_rate': 0.1,
        'model__max_depth': 5,
        'model__min_samples_leaf': 2,
        'model__min_samples_split': 6,
        'model__n_estimators': 500
    }

    # Path to your dataset
    data_path = "regg_data.csv"  # Replace with your data path

    # Run the main workflow
    main(data_path, model_params)