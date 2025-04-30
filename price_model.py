# pylint: disable=missing-module-docstring
import argparse
import os
import glob
import time

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from torcheval.metrics.functional import r2_score
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class StockDataset(Dataset):  # pylint: disable=missing-class-docstring
    def __init__(self,
                 features: np.ndarray,
                 targets: Optional[np.ndarray] = None,
                 data_cols: Optional[list[str]] = None,
                 preprocess: bool = False
                 ) -> None:
        self.features = features
        self.targets = targets

        if self.targets.ndim == 1:
            self.targets = np.expand_dims(self.targets, axis=1)

        if preprocess:
            # find non-numeruc cols and convert
            for col_idx in range(self.features.shape[1]):
                single_feature = self.features[0, col_idx]
                if data_cols is not None:
                    data_cols = list(data_cols)
                    col_name = data_cols[col_idx]
                    if col_name in ['Open', 'Close', 'High', 'Low', 'Date']:
                        # remove row from features and targgets
                        nan_mask = ~np.isnan(self.features)[:, col_idx]
                        self.features = self.features[nan_mask]
                        self.targets = self.targets[nan_mask]

                    old_col = self.features[:, col_idx]

                    if col_name:
                        new_col = self._impute_strategy(old_col, col_name)
                        single_feature = new_col[0]
                    # encode non-numeric cols
                    if isinstance(single_feature, str):
                        le = LabelEncoder()
                        new_col = le.fit_transform(new_col.astype(str))
                        new_col = new_col.astype(float)
                    self.features[:, col_idx] = new_col
            # remove rows where target is nan
            target_nan_mask = ~np.isnan(self.targets).flatten()

            self.targets = self.targets[target_nan_mask]
            self.features = self.features[target_nan_mask]

        # Normalize features
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

        self.features = self.x_scaler.fit_transform(self.features)
        self.targets = self.y_scaler.fit_transform(self.targets)

    def _impute_strategy(self, col: np.ndarray, col_text: str) -> np.ndarray:
        impute_dict = {
            'Date': 'drop',
            'Open': 'drop',
            'Close': 'drop',
            'High': 'drop',
            'Low': 'drop',
            'Volume': 'drop',
            'Dividends': 'zero',
            'Stock Splits': 'zero',
            'MACD': 'zero',
            'MACD_Signal': 'zero',
            'MACD - Signal': 'zero',
            'RSI': 'zero',
            'SMA50': 'median',
            'SMA200': 'median',
            'Breakout': 'zero',
            'Upper_Band': 'forward_backward_fill',
            'Lower_Band': 'forward_backward_fill',
            'Volatility': 'zero',
            'Entry': 'False',
            'Exit': 'False',
            'Trailing_Stop': 'forward_backward_fill',
            'Signal': 'Hold',
            'New_Signal': 'Hold',
            'Sell_Profit_Loss': 'zero',
            'Candlestick_Pattern': 'None',
            'Support': 'zero',
            'Resistance': 'zero',
            'Trend': 'neutral',
            'Trend_trade_signal': 'Hold',
            'Trade_Action': 'hold',
        }
        impute_strategy = impute_dict[col_text]
        if impute_strategy == 'zero':
            col = np.nan_to_num(col, nan=0)
        elif impute_strategy == 'median':
            masked_tensor = col[~np.isnan(col)]
            median = np.median(masked_tensor)
            col = np.where(np.isnan(col), median, col)
        elif impute_strategy == 'forward_backward_fill':
            col = self._forward_backward_fill(col)
        elif impute_strategy == 'None':
            col = np.where(np.isnan(col), 'None', col)
        elif impute_strategy == 'False':
            col = np.where(np.isnan(col), 'False', col)
        elif impute_strategy == 'Hold':
            print(f'Imputing {col_text} with Hold')
            col = np.where(np.isnan(col), 'Hold', col).astype(str)
            print(f'Nans in cols after imputing hold: {sum(pd.isna(col))}')
        elif impute_strategy == 'hold':
            col = np.where(np.isnan(col), 'hold', col)
        return col

    def _forward_backward_fill(self, column: np.ndarray) -> np.ndarray:
        """Perform forward and backward filling for a column."""
        # Forward fill
        for i in range(1, len(column)):
            if np.isnan(column[i]):
                column[i] = column[i - 1]
        # Backward fill
        for i in range(len(column) - 2, -1, -1):
            if np.isnan(column[i]):
                column[i] = column[i + 1]
        return column

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        if self.targets is not None:
            targets = torch.tensor(self.targets[idx], dtype=torch.float32)
            return features, targets
        return features

    def unscale_preds(self, preds: np.ndarray) -> np.ndarray:
        """
        Unscale the predictions using the inverse transform of the scaler.
        """
        return self.y_scaler.inverse_transform(preds)


class StockDataModule(pl.LightningDataModule):
    """
    LightningDataModule for handling stock data.

    Args:
        data_dir (str): The directory path where the CSV files are located.
        target_column (str): The name of the target column in the CSV files. Default is 'Close'.
        batch_size (int): The batch size for data loading. Default is 32.
        preprocess (bool): Whether to preprocess the data. Default is False.
    """

    def __init__(self,
                 data_dir: str,
                 target_column: str = 'Close',
                 batch_size: int = 32,
                 preprocess: bool = False
                 ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.target_column = target_column
        self.batch_size = batch_size
        self.preprocess = preprocess
        self.features_train = None
        self.features_val = None
        self.features_test = None
        self.targets_train = None
        self.targets_val = None
        self.targets_test = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Set up the data for training and testing the price model.

        Args:
            stage (str, optional): The stage of the setup process. Defaults to None.

        Returns:
            None
        """
        start_time = time.time()
        # Load and combine CSVs
        all_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        data_list = [pd.read_csv(file) for file in all_files]
        combined_data = pd.concat(data_list, ignore_index=True)
        data_columns = combined_data.columns

        # Separate features and targets
        features = combined_data.drop(
            columns=['Date', self.target_column], errors='ignore').values
        targets = combined_data[self.target_column].values

        # Handle non-numeric columns in the features
        for col_idx in range(features.shape[1]):
            first_instance = features[0, col_idx]
            if isinstance(first_instance, str):
                le = LabelEncoder()
                features[:, col_idx] = le.fit_transform(features[:, col_idx])

        # Split into training and testing sets
        features_train, self.features_test, targets_train, self.targets_test = train_test_split(
            features.astype(float), targets, test_size=0.2, random_state=42
        )
        split_data = train_test_split(
            features_train.astype(float), targets_train, test_size=0.2, random_state=42
        )
        self.features_train, self.features_val, self.targets_train, self.targets_val = split_data

        # Create datasets
        self.train_dataset = StockDataset(
            self.features_train,
            self.targets_train,
            data_cols=data_columns,
            preprocess=self.preprocess
        )
        self.val_dataset = StockDataset(
            self.features_val,
            self.targets_val,
            data_cols=data_columns,
            preprocess=self.preprocess
        )
        self.test_dataset = StockDataset(
            self.features_test,
            self.targets_test,
            data_cols=data_columns,
            preprocess=self.preprocess
        )
        end_time = time.time()
        print(f"Data setup completed in {end_time - start_time:.2f} seconds")

    def train_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader object for the training dataset.

        Returns:
            DataLoader: A DataLoader object that loads the training dataset in batches.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=6,
            persistent_workers=True,
            shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader object for the validation dataset. Note that this will cause a 
        warning in the autoencoder about passing a val_dataloader without defining a
        validation_step but this can be ignored 

        Returns:
            DataLoader: A DataLoader object for the validation dataset.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=6,
            persistent_workers=True,
            shuffle=False
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader object for the test dataset.

        Returns:
            DataLoader: A DataLoader object for the test dataset.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=6,
            persistent_workers=True,
            shuffle=False
        )


class StockAutoencoder(pl.LightningModule):
    """
    StockAutoencoder is a PyTorch Lightning module that implements an autoencoder for 
    stock data.

    Args:
        input_dim (int): The dimension of the input data.
        hidden_dim (int): The dimension of the hidden layer in the encoder and decoder.
        latent_dim (int): The dimension of the latent space representation.

    Attributes:
        encoder (torch.nn.Sequential): The encoder network.
        decoder (torch.nn.Sequential): The decoder network.
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        """
        Forward pass of the autoencoder.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Reconstructed data.
        """
        for _, layer in enumerate(self.encoder):
            x = layer(x)
        encoded = x
        decoded = self.decoder(encoded)
        return decoded

    def training_step(self,  # pylint: disable=arguments-differ
                      batch: tuple[torch.Tensor, torch.Tensor],
                      batch_idx: int,  # pylint: disable=unused-argument
                      optimizer_idx: int = 0  # pylint: disable=unused-argument
                      ) -> torch.Tensor:
        """
        Training step of the autoencoder.

        Args:
            batch (tuple): A tuple containing the input data and targets.
            batch_idx (int): Index of the current batch.
            optimizer_idx (int): Index of the current optimizer.

        Returns:
            torch.Tensor: The loss value.
        """
        x, _ = batch  # Ignore targets, autoencoder works only on features
        x = x.to('mps')
        reconstructed = self.forward(x)
        loss = torch.nn.functional.mse_loss(reconstructed, x)
        self.log('autoencoder_recon_loss', loss, prog_bar=True)
        return loss

    def validation_step(self,  # pylint: disable=arguments-differ
                        batch: tuple[torch.Tensor, torch.Tensor],
                        batch_idx: int  # pylint: disable=unused-argument
                        ) -> None:
        """
        Validation step of the autoencoder.

        Args:
            batch (tuple): A tuple containing the input data and targets.
            batch_idx (int): Index of the current batch.

        Returns:
            None
        """
        inputs, _ = batch
        reconstructed = self(inputs)
        loss = torch.nn.functional.mse_loss(reconstructed, inputs)
        self.log("val_loss", loss)

    def test_step(self,  # pylint: disable=arguments-differ
                  batch: tuple[torch.Tensor, torch.Tensor],
                  batch_idx: int  # pylint: disable=unused-argument
                  ) -> None:
        """
        Test step of the autoencoder.

        Args:
            batch (tuple): A tuple containing the input data and targets.
            batch_idx (int): Index of the current batch.

        Returns:
            None
        """
        inputs, _ = batch
        reconstructed = self(inputs)
        loss = torch.nn.functional.mse_loss(reconstructed, inputs)
        self.log("test_loss", loss)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer for training.

        Returns:
            torch.optim.Optimizer: The optimizer.
        """
        return torch.optim.Adam(self.parameters(), lr=0.001)


class StockPriceRegressor(pl.LightningModule):
    """
    PyTorch Lightning module for stock price regression.

    Args:
        input_dim (int): The dimension of the input features.

    Attributes:
        model (torch.nn.Sequential): The sequential model for stock price regression.

    """

    def __init__(self, input_dim: int, first_hidden_dim: int) -> None:
        super().__init__()
        self.test_predictions = []
        self.test_targets = []
        self.mse_loss = 0
        second_hidden_dim = first_hidden_dim // 2
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, first_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(input_dim, second_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(second_hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        """ 
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            torch.Tensor: Predicted stock prices.

        """
        return self.model(x)

    def training_step(self,  # pylint: disable=arguments-differ
                      batch: tuple[torch.Tensor, torch.Tensor],
                      batch_idx: int  # pylint: disable=unused-argument
                      ) -> torch.Tensor:
        """
        Training step of the model.

        Args:
            batch (tuple): A tuple containing input features and target values.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The loss value.

        """
        x, y = batch
        x = x.to('mps')
        y = y.to('mps')
        assert not torch.any(torch.isnan(x)), "Features contain NaN!"
        assert not torch.any(torch.isnan(y)), "Targets contain NaN!"
        # features = np.nan_to_num(features)  # Replace NaNs with 0
        # targets = np.nan_to_num(targets)

        # print(f'x: {x}')
        predictions = self.forward(x)
        r2_loss = r2_score(predictions, y)
        # predictions = predictions.squeeze()
        loss = torch.nn.functional.mse_loss(predictions, y)
        self.mse_loss = loss
        self.log('regression_train_mse_loss', loss, prog_bar=True)
        huber_loss = torch.nn.HuberLoss()(predictions, y)
        self.log('regression_train_huber_loss', huber_loss, prog_bar=True)

        self.log('regression_train_r2_loss', r2_loss, prog_bar=True)
        # for name, param in self.named_parameters():
        # if param.grad is not None:
        #     print(f"{name}: {torch.isnan(param.grad).any()}")  # Check for NaNs
        #     print(f"{name} grad: {param.grad}")  # Print gradient values
        return loss

    def validation_step(self,  # pylint: disable=arguments-differ
                        batch: tuple[torch.Tensor, torch.Tensor],
                        batch_idx: int  # pylint: disable=unused-argument
                        ) -> torch.Tensor:
        x, y = batch  # Unpack data
        x = x.to('mps')
        y = y.to('mps')
        predictions = self.model(x)  # Forward pass
        loss = torch.nn.functional.mse_loss(predictions, y)
        r2_loss = r2_score(predictions, y)
        # Log validation loss
        self.log("val_mse_loss", loss, prog_bar=True)
        self.log('val_r2_score', r2_loss, prog_bar=True)
        return loss

    def test_step(self,  # pylint: disable=arguments-differ
                  batch: tuple[torch.Tensor, torch.Tensor],
                  batch_idx: int  # pylint: disable=unused-argument
                  ) -> torch.Tensor:
        x, y = batch
        x = x.to('mps')
        y = y.to('mps')
        predictions = self.model(x)
        loss = torch.nn.functional.mse_loss(predictions, y)
        r2_loss = r2_score(predictions, y)
        # Log test loss
        self.log("test_mse_loss", loss, prog_bar=True)
        self.log('test_r2_score', r2_loss, prog_bar=True)
        self.test_predictions.append(predictions.detach().cpu())
        self.test_targets.append(y.detach().cpu())
        return loss

    def on_test_epoch_start(self) -> None:
        self.test_predictions = []
        self.test_targets = []

    def on_test_epoch_end(self) -> None:
        # Concatenate stored data
        predictions = torch.cat(self.test_predictions, dim=0)
        targets = torch.cat(self.test_targets, dim=0)

        date = pd.to_datetime('today').strftime('%Y-%m-%d')
        # Scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(targets, predictions, alpha=0.7,
                    label="Predictions vs Targets")
        plt.plot([targets.min(), targets.max()], [targets.min(),
                 targets.max()], 'r--', label="Ideal Fit (y = x)")
        plt.xlabel("True Targets")
        plt.ylabel("Predictions")
        plt.title("Scatter Plot: Predictions vs True Targets")
        plt.legend()
        plt.plot([], [], ' ', label=f"MSE: {self.mse_loss:.4f}")
        plt.grid(True)

        # Save or display plot
        plt.savefig(
            f'regressor_preds_vs_targets_epoch_{self.trainer.current_epoch}_{date}.png')

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer for training.

        Returns:
            torch.optim.Optimizer: The optimizer.

        """
        return torch.optim.Adam(self.parameters(), lr=1e-3)


# Define the LSTM model using PyTorch Lightning
class LSTMModel(pl.LightningModule):  # pylint: disable=missing-class-docstring
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_layers: int
                 ) -> None:
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.test_predictions = []
        self.test_targets = []
        # LSTM layer
        self.lstm = torch.nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, self.hidden_size).to(x.device)

        # LSTM output
        out, _ = self.lstm(x, (h0, c0))

        # Take the output from the last time step
        out = self.fc(out)
        return out

    def training_step(self,  # pylint: disable=arguments-differ
                      batch: tuple[torch.Tensor, torch.Tensor],
                      batch_idx: int  # pylint: disable=unused-argument
                      ) -> torch.Tensor:
        x, y = batch
        x = x.to('mps')
        y = y.to('mps')
        y_hat = self.forward(x)
        loss = torch.nn.MSELoss()(y_hat, y)
        self.log("LSTM train_loss", loss)
        return loss

    def validation_step(self,  # pylint: disable=arguments-differ
                        batch: tuple[torch.Tensor, torch.Tensor]
                        ) -> dict[str: torch.Tensor]:
        """
        Perform one step of validation for the given batch.
        Args:
            batch (tuple): A tuple containing the input data and labels.
        Returns:
            dict: A dictionary containing the validation loss and other metrics.
        """
        # Unpack the batch
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        outputs = self.model(inputs)

        loss = self.loss_fn(outputs, targets)

        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == targets).float().mean()
        # loss  = loss.item()
        # acc = accuracy.item()
        self.log('LSTM val loss', loss, prog_bar=True)
        self.log('LSTM val acc', accuracy, prog_bar=True)
        return {'val_loss': loss, 'val_accuracy': accuracy}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def test_step(self,  # pylint: disable=arguments-differ
                  batch: tuple[torch.Tensor, torch.Tensor],
                  batch_idx: int  # pylint: disable=unused-argument
                  ) -> torch.Tensor:
        x, y = batch
        x = x.to('mps')
        y = y.to('mps')
        predictions = self.lstm(x)
        loss = torch.nn.functional.mse_loss(predictions, y)
        r2_loss = r2_score(predictions, y)
        # Log test loss
        self.log("LSTM_test_mse_loss", loss, prog_bar=True)
        self.log('LSTM_test_r2_score', r2_loss, prog_bar=True)
        self.test_predictions.append(predictions.detach().cpu())
        self.test_targets.append(y.detach().cpu())
        return loss

    def on_test_epoch_start(self) -> None:
        self.test_predictions = []
        self.test_targets = []

    def on_test_epoch_end(self) -> None:
        # Concatenate stored data
        predictions = torch.cat(self.test_predictions, dim=0)
        targets = torch.cat(self.test_targets, dim=0)

        date = pd.to_datetime('today').strftime('%Y-%m-%d')
        # Scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(targets, predictions, alpha=0.7,
                    label="Predictions vs Targets")
        plt.plot([targets.min(), targets.max()], [targets.min(),
                 targets.max()], 'r--', label="Ideal Fit (y = x)")
        plt.xlabel("True Targets")
        plt.ylabel("Predictions")
        plt.title("Scatter Plot: Predictions vs True Targets")
        plt.legend()
        plt.plot([], [], ' ', label=f"MSE: {self.mse_loss:.4f}")
        plt.grid(True)

        # Save or display plot
        plt.savefig(
            f'lstm_preds_vs_targets_epoch_{self.trainer.current_epoch}_{date}.png')


def extract_latent_features(
    autoencoder: StockAutoencoder,
    dataloader: DataLoader
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts latent features from the given autoencoder model using the provided dataloader.

    Parameters:
        autoencoder (Autoencoder): The autoencoder model used to extract latent features.
        dataloader (DataLoader): The dataloader containing the data to be processed.

    Returns:
        tuple: A tuple containing the extracted latent features and the corresponding targets.
    """
    start_time = time.time()
    autoencoder.eval()
    latent_features = []
    targets = []
    with torch.no_grad():
        for batch in dataloader:
            features, target = batch
            encoded = autoencoder.encoder(features)
            latent_features.append(encoded.numpy())
            targets.append(target.numpy())
    latent_features = np.vstack(latent_features)
    targets = np.concatenate(targets)
    end_time = time.time()
    print(f'Extracted latent features in {end_time - start_time:.2f} seconds')
    return latent_features, targets


def predict_regressor(regressor: StockPriceRegressor, latent_features: torch.Tensor) -> np.ndarray:
    """
    Predicts the output of a regressor model for a given set of latent features.

    Args:
        regressor (torch.nn.Module): The trained regressor model.
        latent_features (torch.Tensor): The input latent features.

    Returns:
        numpy.ndarray: The predicted outputs of the regressor model.
    """
    start_time = time.time()
    regressor.eval()
    predictions = []
    with torch.no_grad():
        for feature_batch in DataLoader(latent_features, batch_size=128, shuffle=False):
            preds = regressor(feature_batch).numpy()
            predictions.append(preds)
    predictions = np.concatenate(predictions)
    end_time = time.time()
    print(f'Predicted outputs in {end_time - start_time:.2f} seconds')
    return predictions


def compute_metrics(test_targets: np.ndarray, test_predictions: np.ndarray) -> float:
    """
    Compute the Mean Squared Error (MSE) between the test targets and predictions.

    Args:
        test_targets (list): List of test target values.
        test_predictions (list): List of test prediction values.

    Returns:
        float: The computed Mean Squared Error (MSE).
    """
    start_time = time.time()
    test_targets = torch.tensor(test_targets).to('mps')
    test_predictions = torch.tensor(test_predictions).to('mps')
    mse = torch.nn.functional.mse_loss(test_predictions, test_targets)
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    end_time = time.time()
    print(f'Computed metrics in {end_time - start_time:.2f} seconds')
    return mse


def main():  # pylint: disable=missing-function-docstring
    default_data_dir = Path.cwd() / 'signals' / 'sp500'
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=default_data_dir,
                        help='Directory containing the CSV files')
    parser.add_argument('--target_column', type=str,
                        default='Close', help='Target column name')
    parser.add_argument('--ae_batch_size', type=int, default=4096,
                        help='Batch size for training autoencoder')
    parser.add_argument('--re_batch_size', type=int,
                        default=4096, help='Batch size for training regressor')
    parser.add_argument('--ae_hidden_dim', type=int, default=64,
                        help='Autoencoder hidden layer dimension')
    parser.add_argument('--ae_latent_dim', type=int, default=16,
                        help='Autoencoder latent feature dimension')
    parser.add_argument('--ae_max_epochs', type=int, default=20,
                        help='Max epochs for autoencoder training')
    parser.add_argument('--re_max_epochs', type=int, default=100,
                        help='Max epochs for regressor training')
    parser.add_argument('--ls_max_epochs', type=int,
                        default=10, help='Max epochs for LSTM training')
    args = parser.parse_args()
    data_dir = args.data_dir
    target_column = args.target_column
    # batch_size = args.ae_batch_size
    ae_batch_size = args.ae_batch_size
    re_batch_size = args.re_batch_size
    # ae_latent_dim = args.ae_latent_dim
    ae_max_epochs = args.ae_max_epochs
    re_max_epochs = args.re_max_epochs
    ls_max_epochs = args.ls_max_epochs
    hidden_dim = args.ae_hidden_dim
    latent_dim = args.ae_latent_dim

    start_time = time.time()
    # Initialize the DataModule
    print('create stock data module')
    data_module = StockDataModule(
        data_dir, target_column=target_column, batch_size=ae_batch_size, preprocess=True)
    data_module.setup()
    # Get the number of features
    input_dim = data_module.features_train.shape[1]
    print('create autoecnoder')

    # Train the Autoencoder (1m40)
    autoencoder = StockAutoencoder(
        input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    trainer = pl.Trainer(max_epochs=ae_max_epochs, accelerator='mps',
                         devices=1, enable_progress_bar=True, precision=32, profiler="simple")
    print('fitting autoencoder')
    autoencoder_start_time = time.time()
    trainer.fit(autoencoder, data_module)
    autoencoder_end_time = time.time()
    print(
        f'Autoencoder training time: {autoencoder_end_time - autoencoder_start_time:.2f} seconds')

    # Extract Latent Features (27 s)
    print('Extracting latent features for training and test sets')
    latent_train_features, train_targets = extract_latent_features(
        autoencoder, data_module.train_dataloader())
    latent_val_features, val_targets = extract_latent_features(
        autoencoder, data_module.val_dataloader())
    latent_test_features, test_targets = extract_latent_features(
        autoencoder, data_module.test_dataloader())

    # Train Regression Model (30s)
    print('Creating stock dataset (1s)')
    regression_train_dataset = StockDataset(
        latent_train_features, train_targets)
    regression_train_dataloader = DataLoader(
        regression_train_dataset, batch_size=re_batch_size, shuffle=True)
    regression_val_dataset = StockDataset(latent_val_features, val_targets)
    regression_val_dataloader = DataLoader(
        regression_val_dataset, batch_size=re_batch_size, shuffle=False)
    regression_test_dataset = StockDataset(latent_test_features, test_targets)
    regression_test_dataloader = DataLoader(
        regression_test_dataset, batch_size=re_batch_size, shuffle=False)
    print(f'len test dataset: {len(regression_test_dataset)}')
    regressor = StockPriceRegressor(
        input_dim=latent_dim,
        first_hidden_dim=latent_dim
    )
    regression_trainer = pl.Trainer(
        max_epochs=re_max_epochs,
        accelerator='mps',
        devices=1,
        enable_progress_bar=True,
        precision=32,
        profiler='simple'
    )

    # Train the regression model
    print('Training the regression model')
    regression_start_time = time.time()
    regression_trainer.fit(
        regressor, regression_train_dataloader, regression_val_dataloader)
    regression_end_time = time.time()
    print(
        f'Regression model training time: {regression_end_time - regression_start_time:.2f} seconds'
    )
    trainer.test(regressor, regression_test_dataloader)

    # Generate regression predictions  1s
    print(f'shape latent_test_features: {latent_test_features.shape}')
    test_predictions = predict_regressor(regressor, latent_test_features)
    print(f'Len test predictions: {len(test_predictions)}')
    compute_metrics(test_targets, test_predictions)

    # LSTM model
    hidden_size = 20
    output_size = 1
    num_layers = 2

    # Train the model
    lstm_model = LSTMModel(input_size=latent_dim, hidden_size=hidden_size,
                           output_size=output_size, num_layers=num_layers)
    lstm_trainer = pl.Trainer(max_epochs=ls_max_epochs, accelerator='mps',
                              devices=1, enable_progress_bar=True, precision=32, profiler='simple')
    lstm_trainer.fit(lstm_model, regression_train_dataloader)
    lstm_preds = lstm_trainer.test(lstm_model, regression_test_dataloader)
    print(f'LSTM predictions: {lstm_preds}')

    # Unscale predictions and targets
    unscaled_test_preds = data_module.test_dataset.unscale_preds(
        regression_train_dataset.unscale_preds(test_predictions)).tolist()
    unscaled_test_targets = data_module.test_dataset.unscale_preds(
        regression_train_dataset.unscale_preds(test_targets)).tolist()
    end_time = time.time()
    date = pd.to_datetime('today').strftime('%Y-%m-%d')
    prediction_file = Path.cwd() / f'predictions_{date}.csv'
    prediction_file.touch()
    with open(str(prediction_file), 'w', encoding='utf-8') as f:
        f.write('Predictions,Targets\n')
        for pred, target in zip(unscaled_test_preds, unscaled_test_targets):
            f.write(f'{pred[0]},{target[0]}\n')
    print(f'Total time taken: {end_time - start_time:.2f} seconds')


if __name__ == "__main__":
    main()
