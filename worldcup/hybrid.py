from typing import List, Optional, Tuple
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from tensorflow.keras.saving import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime


class Hybrid:
    """Implementation of a SARIMA-LSTM hybrid.
    Worth noting is that this class requires some manual pre-processing and analysis of the data
    to determine the best fitting SARIMA order and LSTM architecture.
    """

    def __init__(
        self,
        sarima_order: Tuple[int, int, int] = (4, 1, 2),
        sarima_seasonal_order: Tuple[int, int, int] = (4, 1, 1, 24),
    ):
        self.sarima: Optional[SARIMAXResults] = None
        self.sarima_order: Tuple[int, int, int] = sarima_order
        self.sarima_seasonal_order: Tuple[int, int, int, int] = sarima_seasonal_order
        self.sarima_residuals: List[float] = []

        self.lstm = None
        self.lstm_look_back: int = 10
        self.lstm_epochs: int = 20
        self.lstm_scaler = None

    def _prepare_lstm_data(self, residuals: List[float]):
        n_lags = self.lstm_look_back
        X, y = [], []
        for i in range(n_lags, len(residuals)):
            X.append(residuals[i - n_lags : i])
            y.append(residuals[i])
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        return X, y

    def _build_default_lstm_model(self, input_shape):
        model = Sequential()
        model.add(
            LSTM(128, return_sequences=True, activation="relu", input_shape=input_shape)
        )
        model.add(Dropout(0.2))
        model.add(LSTM(96))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse")
        return model

    def load_sarima_from_file(self, path: str):
        self.sarima = SARIMAXResults.load(path)

    def load_lstm_from_file(self, path: str):
        self.lstm = load_model(path)

    def _fit_sarima(self, data):
        print("Fitting SARIMA")
        start = datetime.now()

        model = SARIMAX(
            data, order=self.sarima_order, seasonal_order=self.sarima_seasonal_order
        )
        self.sarima = model.fit()

        end = datetime.now()
        print(f"Fit SARIMA in {str(end-start)}")
        self.sarima_residuals = self.sarima.resid

    def _fit_lstm(self):
        """Fits the LSTM on the residuals of the fitted SARIMA model using the lstm_scaler to fit and transform data"""
        if self.sarima_residuals is None:
            raise ValueError("Residuals not found. Fit SARIMA first.")

        print(f"Fitting LSTM")
        start = datetime.now()
        self.lstm_scaler = StandardScaler()

        residuals = self.sarima_residuals.reshape(-1, 1)
        scaled_residuals = self.lstm_scaler.fit_transform(residuals)
        X, y = self._prepare_lstm_data(scaled_residuals)
        model = self._build_default_lstm_model(input_shape=(self.lstm_look_back, 1))
        model.fit(X, y, validation_split=0.2, epochs=self.lstm_epochs)

        end = datetime.now()
        print(f"Fit LSTM in {str(end-start)}")

        self.lstm = model

    def fit(self, data):
        """Fit the hybrid model.
        SARIMA is fit on the data using the orders defined in the constructor.
        A LSTM network is then trained on the residuals of the SARIMA model."""
        self._fit_sarima(data)
        self._fit_lstm()

    def update(self, new_data, arima_refit: bool = False, lstm_refit: bool = True):
        """Updates the ARIMA model with a new observation.
        The refit booleans decides wether or not to update the model weights."""
        self.sarima.append(new_data, refit=arima_refit)
        self.sarima_residuals = self.sarima.resid
        if lstm_refit:
            self._fit_lstm()

    def forecast(self, horizon=1, verbose_return: bool = False):
        if self.sarima is None or self.lstm is None:
            raise ValueError("Fit the model first .fit()")
        """Forecast steps based on horizon. verbose_return decides wether or not to return standalone predictions in a dict"""
        sarima_forecast = self.sarima.forecast(steps=horizon)[0]

        resid_seq = self.sarima_residuals[-self.lstm_look_back :].reshape(-1, 1)
        residuals_scaled = self.lstm_scaler.transform(resid_seq).reshape(
            1, self.lstm_look_back, 1
        )

        lstm_residual_prediction = self.lstm.predict(residuals_scaled)[0][0]

        # Inverse transform of LSTM output
        lstm_residual_prediction = self.lstm_scaler.inverse_transform(
            [[lstm_residual_prediction]]
        )[0][0]

        return {
            "hybrid_forecast": sarima_forecast + lstm_residual_prediction,
            "sarima_forecast": sarima_forecast,
            "lstm_residual_forecast": lstm_residual_prediction,
        }
