"""Classical time series baselines: VARIMA and VAR models using Darts."""

import time
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import VARIMA
from tqdm.auto import tqdm

from utils.config import validate_dataset_name
from utils.dataset import TrafficDataset

# Suppress statsmodels warnings
warnings.filterwarnings("ignore", message=".*Estimation of VARMA.*")
warnings.filterwarnings("ignore", message=".*Maximum Likelihood optimization failed.*")
warnings.filterwarnings("ignore", message=".*Non-stationary starting autoregressive.*")


def interpolate_missing_values(
    continuous_data: np.ndarray, verbose: bool = False
) -> np.ndarray:
    """
    Interpolate missing values (0s) in continuous time series data.

    Assumes 0 values in speed feature (dimension 0) represent missing data
    and interpolates them linearly. Time-of-day feature (dimension 1) is left unchanged.

    Args:
        continuous_data: Array of shape [timesteps, num_nodes, num_features]
        verbose: Print interpolation statistics

    Returns:
        Cleaned array with same shape, 0s interpolated
    """
    cleaned_data = continuous_data.copy()
    num_zeros_total = 0

    num_timesteps, num_nodes, num_features = cleaned_data.shape

    for sensor_id in range(num_nodes):
        # Only interpolate speed feature (dimension 0)
        speed = cleaned_data[:, sensor_id, 0]

        # Count and mark 0s as NaN
        num_zeros = np.sum(speed == 0)
        num_zeros_total += num_zeros

        if num_zeros > 0:
            # Convert to pandas Series for interpolation
            speed_series = pd.Series(speed)
            speed_series[speed_series == 0] = np.nan

            # Linear interpolation
            speed_series = speed_series.interpolate(
                method="linear", limit_direction="both"
            )

            # Fill any remaining NaNs at boundaries
            speed_series = speed_series.bfill().ffill()

            cleaned_data[:, sensor_id, 0] = speed_series.values

    if verbose:
        total_values = num_timesteps * num_nodes
        print(
            f"Interpolated {num_zeros_total:,} zero values "
            f"({num_zeros_total / total_values * 100:.2f}% of speed data)"
        )
        print(f"Average per sensor: {num_zeros_total / num_nodes:.1f}")

    return cleaned_data


def prepare_darts_data_per_sensor(
    continuous_data: np.ndarray,
    start_date: str = "2012-03-01",
    freq: str = "5min",
    speed_only: bool = False,
) -> List[TimeSeries]:
    """
    Prepare per-sensor Darts TimeSeries for VARIMA/ARIMA models.

    Args:
        continuous_data: Array of shape [timesteps, num_nodes, num_features]
        start_date: Start date for time index
        freq: Frequency of observations (default: 5min)
        speed_only: If True, only use speed feature (for ARIMA)

    Returns:
        List of TimeSeries, one per sensor, each with 1 or 2 features
    """
    num_timesteps, num_nodes, num_features = continuous_data.shape

    # Create datetime index
    time_index = pd.date_range(start=start_date, periods=num_timesteps, freq=freq)

    # Create TimeSeries for each sensor
    sensor_series = []
    for sensor_id in range(num_nodes):
        if speed_only:
            # Only use speed feature (dimension 0)
            values = continuous_data[:, sensor_id, 0:1]  # [timesteps, 1]
            columns = ["speed"]
        else:
            # Use both features
            values = continuous_data[:, sensor_id, :]  # [timesteps, 2]
            columns = ["speed", "time_of_day"]

        ts = TimeSeries.from_times_and_values(
            times=time_index,
            values=values,
            columns=columns,
        )
        sensor_series.append(ts)

    return sensor_series


def prepare_darts_data_global(
    continuous_data: np.ndarray,
    start_date: str = "2012-03-01",
    freq: str = "5min",
) -> TimeSeries:
    """
    Prepare global Darts TimeSeries for VAR model.

    Flattens all sensors' speed features into a single multivariate series.
    Only uses speed feature (first feature), ignoring time-of-day.

    Args:
        continuous_data: Array of shape [timesteps, num_nodes, num_features]
        start_date: Start date for time index
        freq: Frequency of observations (default: 5min)

    Returns:
        Single TimeSeries with num_nodes columns (speed only)
    """
    num_timesteps, num_nodes, num_features = continuous_data.shape

    # Create datetime index
    time_index = pd.date_range(start=start_date, periods=num_timesteps, freq=freq)

    # Extract only speed feature (dimension 0)
    speed_data = continuous_data[:, :, 0]  # [timesteps, num_nodes]

    # Create column names
    column_names = [f"sensor_{i}_speed" for i in range(num_nodes)]

    # Create global TimeSeries
    global_ts = TimeSeries.from_times_and_values(
        times=time_index, values=speed_data, columns=column_names
    )

    return global_ts


def fit_and_predict_varima_per_sensor(
    train_sensor_series: List[TimeSeries],
    horizon: int = 12,
    p: int = 2,
    d: int = 1,
    q: int = 1,
    verbose: bool = False,
    n_jobs: int = -1,
    speed_only: bool = False,
) -> np.ndarray:
    """
    Fit VARIMA/ARIMA model per sensor and generate predictions immediately.

    Does NOT store models in memory - fits, predicts, and discards each model
    to avoid memory issues with large datasets.

    Args:
        train_sensor_series: List of TimeSeries, one per sensor
        horizon: Number of steps to predict
        p: Autoregressive order
        d: Differencing order
        q: Moving average order
        verbose: Print progress
        n_jobs: Number of parallel jobs (-1 for all cores, 1 for sequential)
        speed_only: If True, models are univariate ARIMA (1 feature per sensor)

    Returns:
        Predictions array of shape [1, horizon, num_sensors, 1]
    """
    from joblib import Parallel, delayed

    num_sensors = len(train_sensor_series)
    model_name = "ARIMA" if speed_only else "VARIMA"

    def fit_and_predict_single_sensor(sensor_id, ts):
        """Fit VARIMA/ARIMA model and predict for a single sensor, then discard model."""
        import warnings

        from darts.models import ARIMA

        # Suppress warnings in each worker process
        warnings.filterwarnings("ignore", message=".*Estimation of VARMA.*")
        warnings.filterwarnings(
            "ignore", message=".*Maximum Likelihood optimization failed.*"
        )
        warnings.filterwarnings(
            "ignore", message=".*Non-stationary starting autoregressive.*"
        )

        try:
            # Use ARIMA for univariate, VARIMA for multivariate
            if speed_only:
                model = ARIMA(p=p, d=d, q=q)
            else:
                model = VARIMA(p=p, d=d, q=q)

            model.fit(ts)

            # Generate prediction
            pred = model.predict(n=horizon)
            pred_values = pred.values()  # Shape: [horizon, num_features]

            # Extract speed only
            if speed_only or pred_values.shape[1] == 1:
                # Already univariate or speed-only
                speed_pred = (
                    pred_values if pred_values.shape[1] == 1 else pred_values[:, 0:1]
                )
            else:
                # Extract first feature (speed) from bivariate
                speed_pred = pred_values[:, 0:1]  # Shape: [horizon, 1]

            # Model is discarded here, only prediction returned
            return speed_pred

        except Exception:
            if verbose:
                print(f"\n  Warning: Sensor {sensor_id} failed")
            return np.full((horizon, 1), np.nan)

    if verbose:
        print(
            f"Fitting and predicting {num_sensors} {model_name} models (n_jobs={n_jobs})..."
        )

    # Process sensors in parallel
    predictions_list = Parallel(
        n_jobs=n_jobs, verbose=0, batch_size=1, pre_dispatch="2*n_jobs"
    )(
        delayed(fit_and_predict_single_sensor)(i, train_sensor_series[i])
        for i in tqdm(
            range(num_sensors), desc=f"{model_name} fit+predict", disable=not verbose
        )
    )

    # Stack predictions: [num_sensors, horizon, 1] -> [1, horizon, num_sensors, 1]
    predictions = np.stack(predictions_list, axis=0)  # [num_sensors, horizon, 1]
    predictions = np.expand_dims(
        predictions.transpose(1, 0, 2), axis=0
    )  # [1, horizon, num_sensors, 1]

    num_successful = np.sum(~np.isnan(predictions[0, 0, :, 0]))
    if verbose:
        print(f"Successfully predicted for {num_successful}/{num_sensors} sensors")

    return predictions


def fit_var_global(
    train_global_ts: TimeSeries,
    maxlags: int = 12,
    verbose: bool = False,
):
    """
    Fit global VAR model using statsmodels directly (OLS, much faster than MLE).

    Args:
        train_global_ts: Global TimeSeries with all sensors (speed only)
        maxlags: Maximum number of lags (VAR order p)
        verbose: Print progress

    Returns:
        Fitted statsmodels VARResults object
    """
    from statsmodels.tsa.api import VAR

    if verbose:
        data_shape = train_global_ts.values().shape
        num_timesteps, num_sensors = data_shape
        num_params = num_sensors * num_sensors * maxlags
        print(f"Fitting VAR model with maxlags={maxlags} using OLS (fast)")
        print(f"Data shape: {data_shape}")
        print(f"Parameters to estimate: ~{num_params:,}")

    # Convert to pandas DataFrame for statsmodels
    df = train_global_ts.to_dataframe()

    # Create and fit VAR model (uses OLS, not MLE)
    model = VAR(df)

    start_time = time.time()
    fitted_model = model.fit(maxlags=maxlags, verbose=verbose)
    elapsed = time.time() - start_time

    if verbose:
        print(f"VAR fitted in {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")
        print(f"Selected lag order: {fitted_model.k_ar}")

    return fitted_model


# predict_varima removed - now done in fit_and_predict_varima_per_sensor


def predict_var(
    model,
    horizon: int = 12,
    num_samples: int = 1,
    verbose: bool = False,
) -> np.ndarray:
    """
    Generate predictions from statsmodels VAR model.

    Args:
        model: Fitted statsmodels VARResults object
        horizon: Number of steps to predict
        num_samples: Number of samples to predict
        verbose: Print progress

    Returns:
        Predictions array of shape [num_samples, horizon, num_sensors, 1]
    """
    if verbose:
        print(f"Generating {num_samples} predictions with horizon={horizon}")

    # Forecast using statsmodels VAR
    # Need the last k_ar observations for forecasting
    forecast = model.forecast(model.endog[-model.k_ar :], steps=horizon)
    # forecast shape: [horizon, num_sensors]

    num_sensors = forecast.shape[1]

    # Initialize predictions array
    predictions = np.zeros((num_samples, horizon, num_sensors, 1))

    # Replicate prediction for all samples (stationary assumption)
    for sample_idx in range(num_samples):
        predictions[sample_idx, :, :, 0] = forecast

    if verbose:
        print(f"Generated predictions with shape: {predictions.shape}")

    return predictions


def get_varima_predictions(
    dataset_name: str,
    dataset: Dict[str, TrafficDataset],
    split: str = "test",
    p: int = 2,
    d: int = 1,
    q: int = 1,
    force_refit: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """
    Get VARIMA predictions for a dataset.

    Fits per-sensor VARIMA models on training data and generates predictions
    for the specified split. Models are NOT stored - fit, predict, discard.

    Args:
        dataset_name: Name of dataset (e.g., "METR-LA")
        dataset: Dictionary of TrafficDataset splits
        split: Split to predict on (default: "test")
        p: VARIMA autoregressive order
        d: VARIMA differencing order
        q: VARIMA moving average order
        force_refit: Unused (kept for API compatibility)
        verbose: Print progress

    Returns:
        Predictions array of shape [num_samples, horizon, num_nodes, 1]

    Note:
        Only generates predictions for a single sample (assumes stationary process).
        For multiple samples, predictions are replicated.
    """
    dataset_name = validate_dataset_name(dataset_name)

    if verbose:
        print("Fitting VARIMA models on training data...")

    # Get continuous training data
    train_continuous = dataset["train"].to_continuous(include_targets=True)

    # Interpolate missing values
    train_continuous_clean = interpolate_missing_values(
        train_continuous, verbose=verbose
    )

    # Prepare per-sensor TimeSeries (bivariate: speed + time_of_day)
    train_sensor_series = prepare_darts_data_per_sensor(
        train_continuous_clean, speed_only=False
    )

    # Get prediction parameters
    horizon = dataset[split].horizon
    num_samples = len(dataset[split])

    if verbose:
        print(f"Generating predictions for horizon={horizon}")

    # Fit and predict (no models stored in memory)
    start_time = time.time()
    predictions = fit_and_predict_varima_per_sensor(
        train_sensor_series,
        horizon=horizon,
        p=p,
        d=d,
        q=q,
        verbose=verbose,
        speed_only=False,
    )
    end_time = time.time()
    fitting_time = end_time - start_time

    if verbose:
        print(
            f"Fit+predict completed in {fitting_time:.2f} seconds ({fitting_time / 60:.2f} minutes)"
        )

    # Replicate predictions for all samples (assumes stationary)
    if num_samples > 1:
        predictions = np.repeat(predictions, num_samples, axis=0)

    return predictions


def get_arima_predictions(
    dataset_name: str,
    dataset: Dict[str, TrafficDataset],
    split: str = "test",
    p: int = 2,
    d: int = 1,
    q: int = 1,
    force_refit: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """
    Get ARIMA predictions for a dataset (speed only, univariate per sensor).

    Fits per-sensor ARIMA models on training data and generates predictions
    for the specified split. Models are NOT stored - fit, predict, discard.

    Args:
        dataset_name: Name of dataset (e.g., "METR-LA")
        dataset: Dictionary of TrafficDataset splits
        split: Split to predict on (default: "test")
        p: ARIMA autoregressive order
        d: ARIMA differencing order
        q: ARIMA moving average order
        force_refit: Unused (kept for API compatibility)
        verbose: Print progress

    Returns:
        Predictions array of shape [num_samples, horizon, num_nodes, 1]

    Note:
        Only generates predictions for a single sample (assumes stationary process).
        For multiple samples, predictions are replicated.
    """
    dataset_name = validate_dataset_name(dataset_name)

    if verbose:
        print("Fitting ARIMA models on training data (speed only)...")

    # Get continuous training data
    train_continuous = dataset["train"].to_continuous(include_targets=True)

    # Interpolate missing values
    train_continuous_clean = interpolate_missing_values(
        train_continuous, verbose=verbose
    )

    # Prepare per-sensor TimeSeries (univariate: speed only)
    train_sensor_series = prepare_darts_data_per_sensor(
        train_continuous_clean, speed_only=True
    )

    # Get prediction parameters
    horizon = dataset[split].horizon
    num_samples = len(dataset[split])

    if verbose:
        print(f"Generating predictions for horizon={horizon}")

    # Fit and predict (no models stored in memory)
    start_time = time.time()
    predictions = fit_and_predict_varima_per_sensor(
        train_sensor_series,
        horizon=horizon,
        p=p,
        d=d,
        q=q,
        verbose=verbose,
        speed_only=True,
    )
    end_time = time.time()
    fitting_time = end_time - start_time

    if verbose:
        print(
            f"Fit+predict completed in {fitting_time:.2f} seconds ({fitting_time / 60:.2f} minutes)"
        )

    # Replicate predictions for all samples (assumes stationary)
    if num_samples > 1:
        predictions = np.repeat(predictions, num_samples, axis=0)

    return predictions


def get_ar_predictions(
    dataset_name: str,
    dataset: Dict[str, TrafficDataset],
    split: str = "test",
    p: int = 12,
    force_refit: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """
    Get AR predictions for a dataset (speed only, univariate per sensor).

    AR is ARIMA with d=0, q=0. Uses same lag order as VAR for fair comparison
    in ablation study (univariate vs multivariate).

    Args:
        dataset_name: Name of dataset (e.g., "METR-LA")
        dataset: Dictionary of TrafficDataset splits
        split: Split to predict on (default: "test")
        p: AR order (default: 12 to match VAR)
        force_refit: Unused (kept for API compatibility)
        verbose: Print progress

    Returns:
        Predictions array of shape [num_samples, horizon, num_nodes, 1]

    Note:
        Only generates predictions for a single sample (assumes stationary process).
        For multiple samples, predictions are replicated.
    """
    dataset_name = validate_dataset_name(dataset_name)

    if verbose:
        print(f"Fitting AR({p}) models on training data (speed only)...")

    # Get continuous training data
    train_continuous = dataset["train"].to_continuous(include_targets=True)

    # Interpolate missing values
    train_continuous_clean = interpolate_missing_values(
        train_continuous, verbose=verbose
    )

    # Prepare per-sensor TimeSeries (univariate: speed only)
    train_sensor_series = prepare_darts_data_per_sensor(
        train_continuous_clean, speed_only=True
    )

    # Get prediction parameters
    horizon = dataset[split].horizon
    num_samples = len(dataset[split])

    if verbose:
        print(f"Generating predictions for horizon={horizon}")

    # Fit and predict with d=0, q=0 (pure AR model)
    start_time = time.time()
    predictions = fit_and_predict_varima_per_sensor(
        train_sensor_series,
        horizon=horizon,
        p=p,
        d=0,
        q=0,
        verbose=verbose,
        speed_only=True,
    )
    end_time = time.time()
    fitting_time = end_time - start_time

    if verbose:
        print(
            f"Fit+predict completed in {fitting_time:.2f} seconds ({fitting_time / 60:.2f} minutes)"
        )

    # Replicate predictions for all samples (assumes stationary)
    if num_samples > 1:
        predictions = np.repeat(predictions, num_samples, axis=0)

    return predictions


def get_var_predictions(
    dataset_name: str,
    dataset: Dict[str, TrafficDataset],
    split: str = "test",
    maxlags: int = 12,
    force_refit: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """
    Get VAR predictions for a dataset.

    Fits a global VAR model on all sensors jointly using training data (speed only)
    and generates predictions for the specified split.

    Args:
        dataset_name: Name of dataset (e.g., "METR-LA")
        dataset: Dictionary of TrafficDataset splits
        split: Split to predict on (default: "test")
        maxlags: VAR maximum lags
        force_refit: Unused (kept for API compatibility)
        verbose: Print progress

    Returns:
        Predictions array of shape [num_samples, horizon, num_nodes, 1]
    """
    dataset_name = validate_dataset_name(dataset_name)

    if verbose:
        print("Fitting VAR model on training data (speed only)...")

    # Get continuous training data
    train_continuous = dataset["train"].to_continuous(include_targets=True)

    # Interpolate missing values
    train_continuous_clean = interpolate_missing_values(
        train_continuous, verbose=verbose
    )

    # Prepare global TimeSeries (speed only)
    train_global_ts = prepare_darts_data_global(train_continuous_clean)

    # Fit model with timing
    start_time = time.time()
    model = fit_var_global(train_global_ts, maxlags=maxlags, verbose=verbose)
    end_time = time.time()
    fitting_time = end_time - start_time

    if verbose:
        print(
            f"Model fitting completed in {fitting_time:.2f} seconds ({fitting_time / 60:.2f} minutes)"
        )

    # Generate predictions
    num_samples = len(dataset[split])
    horizon = dataset[split].horizon

    predictions = predict_var(
        model, horizon=horizon, num_samples=num_samples, verbose=verbose
    )

    return predictions
