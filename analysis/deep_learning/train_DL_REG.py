def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os.path
import os
import joblib
from datetime import datetime

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score


# Define a PyTorch Dataset class
class CarPriceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Define an even deeper Neural Network model for regression
# Define an extremely deep Neural Network with DenseNet-inspired architecture
# Define an extremely deep Neural Network with DenseNet-inspired architecture
class UltraDeepRegressionNN(nn.Module):
    def __init__(self, input_size):
        super(UltraDeepRegressionNN, self).__init__()

        # Initial feature extraction
        self.input_block = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        # Dense Block 1 (512 features)
        self.dense_block1_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512 + i * 128, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.25)
            ) for i in range(4)
        ])

        # Transition Layer 1
        self.transition1 = nn.Sequential(
            nn.Linear(1024, 512),  # 512 + 4*128 = 1024
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Dense Block 2 (512 features)
        self.dense_block2_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512 + i * 128, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.25)
            ) for i in range(4)
        ])

        # Transition Layer 2
        self.transition2 = nn.Sequential(
            nn.Linear(1024, 384),  # 512 + 4*128 = 1024
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Dense Block 3 (384 features)
        self.dense_block3_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(384 + i * 96, 96),
                nn.BatchNorm1d(96),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.25)
            ) for i in range(3)
        ])

        # Transition Layer 3
        self.transition3 = nn.Sequential(
            nn.Linear(672, 256),  # 384 + 3*96 = 672
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Dense Block 4 (256 features)
        self.dense_block4_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256 + i * 64, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.25)
            ) for i in range(3)
        ])

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(448, 128),  # 256 + 3*64 = 448
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        # Output layer with custom initialization
        self.output_layer = nn.Linear(32, 1)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m == self.output_layer:
                    # Special initialization for output layer
                    nn.init.xavier_normal_(m.weight)
                    nn.init.zeros_(m.bias)
                else:
                    # He initialization for most layers
                    nn.init.kaiming_normal_(m.weight, mode='fan_in',
                                            nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x):
        # Initial feature extraction
        x = self.input_block(x)

        # Dense Block 1
        features = [x]
        for layer in self.dense_block1_layers:
            new_features = layer(torch.cat(features, dim=1))
            features.append(new_features)
        x = torch.cat(features, dim=1)

        # Transition 1
        x = self.transition1(x)

        # Dense Block 2
        features = [x]
        for layer in self.dense_block2_layers:
            new_features = layer(torch.cat(features, dim=1))
            features.append(new_features)
        x = torch.cat(features, dim=1)

        # Transition 2
        x = self.transition2(x)

        # Dense Block 3
        features = [x]
        for layer in self.dense_block3_layers:
            new_features = layer(torch.cat(features, dim=1))
            features.append(new_features)
        x = torch.cat(features, dim=1)

        # Transition 3
        x = self.transition3(x)

        # Dense Block 4
        features = [x]
        for layer in self.dense_block4_layers:
            new_features = layer(torch.cat(features, dim=1))
            features.append(new_features)
        x = torch.cat(features, dim=1)

        # Bottleneck
        x = self.bottleneck(x)

        # Attention mechanism
        attention_weights = self.attention(x)
        x = x * attention_weights

        # Output layer
        x = self.output_layer(x)

        return x


# Update the CarPriceRegressor class to use the new UltraDeepRegressionNN model
class CarPriceRegressor(object):
    def __init__(self):
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_r2': [],
            'val_r2': [],
            'lr': []
        }
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def fit(self, X_train, y_train, X_val, y_val, epochs=200, batch_size=16, learning_rate=0.001):
        # Scale the features
        X_train_scaled = pd.DataFrame(
            self.scaler_X.fit_transform(X_train),
            columns=X_train.columns
        )
        y_train_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()

        X_val_scaled = pd.DataFrame(
            self.scaler_X.transform(X_val),
            columns=X_val.columns
        )
        y_val_scaled = self.scaler_y.transform(y_val.values.reshape(-1, 1)).flatten()

        # Create datasets and dataloaders
        train_dataset = CarPriceDataset(X_train_scaled, y_train_scaled)
        val_dataset = CarPriceDataset(X_val_scaled, y_val_scaled)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Initialize the ultra deep model
        input_size = X_train.shape[1]
        self.model = UltraDeepRegressionNN(input_size)

        # Move model to device (GPU if available)
        self.model = self.model.to(self.device)

        # Print model architecture
        print(f"Model Architecture:")
        print(self.model)
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters())}")

        # Define loss function and optimizer with weight decay
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)

        # Learning rate scheduler - more sophisticated with warmup
        def lr_lambda(epoch):
            warmup_epochs = 10
            if epoch < warmup_epochs:
                # Linear warmup
                return epoch / warmup_epochs
            else:
                # Cosine annealing
                return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # Early stopping parameters
        best_val_loss = float('inf')
        patience = 350
        patience_counter = 0
        best_model_state = None

        # Training loop
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_preds = []
            train_true = []

            for X_batch, y_batch in train_loader:
                # Move data to device
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)

                # Add L1 regularization for additional sparsity
                l1_lambda = 1e-5
                l1_norm = sum(p.abs().sum() for p in self.model.parameters())
                loss = loss + l1_lambda * l1_norm

                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item()
                train_preds.extend(outputs.detach().cpu().numpy())
                train_true.extend(y_batch.cpu().numpy())

            # Calculate metrics for training set
            train_loss = train_loss / len(train_loader)
            train_preds = np.array(train_preds).flatten()
            train_true = np.array(train_true).flatten()
            train_r2 = r2_score(train_true, train_preds)

            # Validation
            self.model.eval()
            val_loss = 0.0
            val_preds = []
            val_true = []

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    # Move data to device
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)

                    val_loss += loss.item()
                    val_preds.extend(outputs.cpu().numpy())
                    val_true.extend(y_batch.cpu().numpy())

            # Calculate metrics for validation set
            val_loss = val_loss / len(val_loader)
            val_preds = np.array(val_preds).flatten()
            val_true = np.array(val_true).flatten()
            val_r2 = r2_score(val_true, val_preds)

            # Learning rate scheduler step
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            # Save metrics to history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_r2'].append(train_r2)
            self.history['val_r2'].append(val_r2)
            self.history['lr'].append(current_lr)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict()
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                # Load the best model
                self.model.load_state_dict(best_model_state)
                break

            # Print progress
            if (epoch + 1) % 10 == 0:
                print(
                    f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                    f'Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}, LR: {current_lr:.6f}'
                )

        # If early stopping didn't trigger, ensure we use the best model
        if best_model_state is not None and patience_counter < patience:
            self.model.load_state_dict(best_model_state)
            print("Loaded best model based on validation loss")

        return self

    def predict(self, X):
        # Scale the features
        X_scaled = pd.DataFrame(
            self.scaler_X.transform(X),
            columns=X.columns
        )

        # Create dataset and move to device
        test_tensor = torch.tensor(X_scaled.values, dtype=torch.float32).to(self.device)

        # Make predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(test_tensor)

        # Move back to CPU for numpy conversion
        predictions = predictions.cpu()

        # Inverse transform the predictions
        predictions = self.scaler_y.inverse_transform(predictions.numpy())

        return predictions.flatten()

    def evaluate(self, X, y):
        predictions = self.predict(X)

        # Calculate metrics
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)
        explained_var = explained_variance_score(y, predictions)

        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'Explained Variance': explained_var
        }

        # Create a scatter plot of actual vs. predicted values
        plt.figure(figsize=(12, 8))
        plt.scatter(y, predictions, alpha=0.6, c='blue', edgecolors='w', s=70)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
        plt.xlabel('Actual Prices', fontsize=14)
        plt.ylabel('Predicted Prices', fontsize=14)
        plt.title('Actual vs. Predicted Car Prices', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('prediction_scatter.png', dpi=300)
        plt.show()

        # Create a residual plot
        residuals = y - predictions
        plt.figure(figsize=(12, 8))
        plt.scatter(predictions, residuals, alpha=0.6, c='green', edgecolors='w', s=70)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
        plt.xlabel('Predicted Prices', fontsize=14)
        plt.ylabel('Residuals', fontsize=14)
        plt.title('Residual Plot', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('residual_plot.png', dpi=300)
        plt.show()

        return metrics

    def plot_metrics(self):
        plt.figure(figsize=(18, 15))

        # Plot loss
        plt.subplot(3, 1, 1)
        plt.plot(self.history['train_loss'], label='Training Loss', color='blue', linewidth=2)
        plt.plot(self.history['val_loss'], label='Validation Loss', color='red', linewidth=2)
        plt.title('Loss over Epochs', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss (MSE)', fontsize=14)
        plt.yscale('log')  # Log scale often helps visualize loss curves better
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        # Plot R² score
        plt.subplot(3, 1, 2)
        plt.plot(self.history['train_r2'], label='Training R²', color='blue', linewidth=2)
        plt.plot(self.history['val_r2'], label='Validation R²', color='red', linewidth=2)
        plt.title('R² over Epochs', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('R²', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        # Plot learning rate
        plt.subplot(3, 1, 3)
        plt.plot(self.history['lr'], color='green', linewidth=2)
        plt.title('Learning Rate Schedule', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Learning Rate', fontsize=14)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_metrics.png', dpi=300)
        plt.show()

    def save(self, filepath):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Move model to CPU for saving
        self.model = self.model.to('cpu')

        # Save the model state, scalers, and history
        model_dict = {
            'model_state': self.model.state_dict(),
            'model_architecture': 'UltraDeepRegressionNN',  # Updated architecture info
            'input_size': self.model.input_block[0].in_features,  # Updated access pattern
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'history': self.history
        }

        torch.save(model_dict, filepath)
        print(f"Model saved to {filepath}")

        # Move model back to original device
        self.model = self.model.to(self.device)

    @classmethod
    def load(cls, filepath):
        # Load the model state, scalers, and history
        model_dict = torch.load(filepath, map_location=torch.device('cpu'))

        # Create a new instance of the regressor
        regressor = cls()

        # Create and load the model based on architecture
        input_size = model_dict['input_size']
        if model_dict.get('model_architecture') == 'UltraDeepRegressionNN':
            regressor.model = UltraDeepRegressionNN(input_size)

        regressor.model.load_state_dict(model_dict['model_state'])

        # Move model to appropriate device
        regressor.model = regressor.model.to(regressor.device)

        # Load the scalers and history
        regressor.scaler_X = model_dict['scaler_X']
        regressor.scaler_y = model_dict['scaler_y']
        regressor.history = model_dict['history']

        return regressor


def train_DL_REG(X_train, X_val, y_train, y_val, X_test, y_test):
    # Initialize the regressor
    regressor = CarPriceRegressor()

    # Train the model with more stable hyperparameters
    print("Training ultra-deep neural network model...")

    # Start with a smaller learning rate and batch size to ensure stability
    regressor.fit(X_train, y_train, X_val, y_val, epochs=200, batch_size=32, learning_rate=0.002)

    # Evaluate on test set
    test_metrics = regressor.evaluate(X_test, y_test)
    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    # Plot the training metrics
    regressor.plot_metrics()

    # Save the model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"../../DL_models/car_price_ultra_deep_regressor_{timestamp}.pt"
    regressor.save(model_path)

    return regressor


def train_DL_REG(X_train, X_val, y_train, y_val, X_test, y_test):
    # Initialize the regressor
    regressor = CarPriceRegressor()

    # Train the model
    print("Training ultra-deep neural network model...")
    regressor.fit(X_train, y_train, X_val, y_val, epochs=500, batch_size=64, learning_rate=0.003)

    # Evaluate on test set
    test_metrics = regressor.evaluate(X_test, y_test)
    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    # Plot the training metrics
    regressor.plot_metrics()

    # Save the model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"../../DL_models/car_price_ultra_deep_regressor_{timestamp}.pt"
    regressor.save(model_path)

    return regressor


def main():
    if (os.path.exists("../../data/train_test_validate_data/train_test_data.csv")):
        data = pd.read_csv("../../data/train_test_validate_data/train_test_data.csv")
    else:
        ownership_df_class = pd.read_csv("../../data/final_data/final_ownership_data.csv")
        sales_df_reg = pd.read_csv("../../data/final_data/final_sales_data.csv")

        ownership_df_class = ownership_df_class.drop(columns=['Unnamed: 0', '_c0'])
        sales_df_reg = sales_df_reg.drop(columns=['Unnamed: 0', '_c0'])
        sales_df_reg = sales_df_reg[['Annual_Income', 'Price']]

        def find_closest_income_match(target_income, reference_df):
            closest_income_idx = (reference_df['Annual_Income'] - target_income).abs().idxmin()
            return reference_df.loc[closest_income_idx, 'Price']

        ownership_df_class['Price'] = ownership_df_class.apply(
            lambda row: find_closest_income_match(row['Annual_Income'], sales_df_reg)
            if row['Car'] == 'Yes'
            else 0,
            axis=1
        )

        ownership_sales = ownership_df_class.__deepcopy__()
        ownership_sales.to_csv(path_or_buf="../../data/train_test_validate_data/train_test_data.csv")
        data = pd.read_csv("../../data/train_test_validate_data/train_test_data.csv")

    data_reg = data.__deepcopy__()

    data_reg = data_reg.where(cond=(data_reg['Car'] == 'Yes')).dropna(axis=0).drop(
        columns=['Unnamed: 0', 'Car', 'Occupation'])

    label_mapping = {
        'Unknown': 0,
        'Unstable': 1,
        'Fair': 2,
        'Stable': 3,
        'Good': 4,
        'Excellent': 5
    }

    data_reg['Finance_Status'] = data_reg['Finance_Status'].replace(label_mapping)
    data_reg = data_reg.where((data_reg['Annual_Income'] < 45000)).dropna(axis=0)

    data_reg['Annual_Income_Credit_Score'] = data_reg['Annual_Income'] / data_reg['Credit_Score']
    data_reg['Annual_Income_Years_Employment'] = data_reg['Annual_Income'] / data_reg['Years_of_Employment']
    data_reg['Years_Employment_Credit_Score'] = data_reg['Credit_Score'] / data_reg['Years_of_Employment']

    Y = data_reg['Price']
    X = data_reg.drop(columns=['Price'])

    X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Create model directories if they don't exist
    os.makedirs("../../DL_models", exist_ok=True)

    # Train the model
    regressor = train_DL_REG(X_train, X_val, y_train, y_val, X_test, y_test)


if __name__ == '__main__':
    # Set up PyTorch device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    main()
    print("Done")